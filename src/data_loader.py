"""
data_loader.py — Handles loading, cleaning, and validating the inventory Excel dataset.

This module provides robust parsing of the DesiCrew Inventory-Records-Sample-Data.xlsx
file, which has a non-standard layout (title rows, blank first column, multi-line headers).
It also supports arbitrary user-uploaded Excel files with standard headers.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path


# Canonical column names for the supplied inventory dataset
CANONICAL_COLUMNS = [
    "Product ID",
    "Product Name",
    "Opening Stock",
    "Purchase/Stock In",
    "Number of Units Sold",
    "Hand-In-Stock",
    "Cost Price Per Unit (USD)",
    "Cost Price Total (USD)",
]

# Map raw (possibly multi-line) Excel header text to canonical names
_HEADER_MAP = {
    "product id": "Product ID",
    "product name": "Product Name",
    "opening stock": "Opening Stock",
    "opening \nstock": "Opening Stock",
    "purchase/stock in": "Purchase/Stock In",
    "purchase/\nstock in": "Purchase/Stock In",
    "number of units sold": "Number of Units Sold",
    "number of \nunits sold": "Number of Units Sold",
    "hand-in-stock": "Hand-In-Stock",
    "hand-in-\nstock": "Hand-In-Stock",
    "cost price per unit (usd)": "Cost Price Per Unit (USD)",
    "cost price \nper unit (usd)": "Cost Price Per Unit (USD)",
    "cost price total (usd)": "Cost Price Total (USD)",
    "cost price\ntotal (usd)": "Cost Price Total (USD)",
}


def _normalise_header(raw: str) -> str:
    """Normalise a raw header string to its canonical form."""
    if not isinstance(raw, str):
        return str(raw)
    key = raw.strip().lower()
    return _HEADER_MAP.get(key, raw.strip())


def load_dataset(
    file_path_or_buffer,
    filename: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Universal dataset loader for any tabular dataset document:
    - Excel (.xlsx, .xls) with non-standard or standard headers
    - CSV (.csv), TSV (.tsv) with various encodings and delimiters

    Parameters
    ----------
    file_path_or_buffer : str, Path, or file-like buffer
        Path or buffer of the dataset.
    filename : str, optional
        Filename hint to assist format detection.

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame.
    meta : dict
        Metadata including raw_shape, final_shape, columns, cleaning_notes.
    """
    meta: Dict[str, Any] = {"cleaning_notes": []}

    # Determine file name / format
    name = filename or ""
    if not name and hasattr(file_path_or_buffer, "name"):
        name = file_path_or_buffer.name
    elif not name and isinstance(file_path_or_buffer, (str, Path)):
        name = Path(file_path_or_buffer).name
    name_lower = str(name).lower()

    is_csv = name_lower.endswith(".csv") or name_lower.endswith(".tsv") or name_lower.endswith(".txt")

    if is_csv:
        # --- Handle CSV / TSV ---
        df = None
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                if hasattr(file_path_or_buffer, "seek"):
                    file_path_or_buffer.seek(0)
                df = pd.read_csv(file_path_or_buffer, sep=None, engine="python", encoding=enc)
                meta["cleaning_notes"].append(f"Loaded CSV with {enc} encoding.")
                break
            except Exception:
                continue

        if df is None:
            if hasattr(file_path_or_buffer, "seek"):
                file_path_or_buffer.seek(0)
            df = pd.read_csv(file_path_or_buffer, encoding="utf-8", on_bad_lines="skip")
            meta["cleaning_notes"].append("Loaded CSV with utf-8 fallback.")

        meta["raw_shape"] = df.shape

        # Clean column names
        df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

        # Drop entirely empty columns
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            meta["cleaning_notes"].append(f"Dropped {len(empty_cols)} empty column(s).")

        # Normalise canonical headers if present
        df.columns = [_normalise_header(c) for c in df.columns]

        # Drop entirely empty rows
        df = df.dropna(how="all").reset_index(drop=True)

        # Coerce numeric columns
        for c in df.columns:
            if df[c].dtype == object:
                sample = df[c].dropna().head(20).astype(str)
                cleaned_sample = sample.str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
                if not cleaned_sample.empty and pd.to_numeric(cleaned_sample, errors="coerce").notna().mean() > 0.7:
                    df[c] = pd.to_numeric(
                        df[c].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip(),
                        errors="coerce"
                    )

        meta["final_shape"] = df.shape
        meta["columns"] = list(df.columns)
        return df, meta

    # --- Handle Excel ---
    raw = None
    try:
        if hasattr(file_path_or_buffer, "seek"):
            file_path_or_buffer.seek(0)
        raw = pd.read_excel(file_path_or_buffer, header=None)
        meta["raw_shape"] = raw.shape
        header_row = _detect_header_row(raw)
        meta["detected_header_row"] = header_row

        if header_row is None:
            header_row = 0
            meta["cleaning_notes"].append("Using first row as header.")

        headers = raw.iloc[header_row].tolist()
        df = raw.iloc[header_row + 1 :].copy()
        df.columns = headers
        df = df.reset_index(drop=True)
    except Exception:
        if hasattr(file_path_or_buffer, "seek"):
            file_path_or_buffer.seek(0)
        df = pd.read_excel(file_path_or_buffer)
        meta["raw_shape"] = df.shape

    # Drop fully-empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        meta["cleaning_notes"].append(f"Dropped {len(empty_cols)} entirely empty column(s).")

    # Drop columns whose header is NaN
    nan_header_cols = [c for c in df.columns if (isinstance(c, float) and np.isnan(c))]
    if nan_header_cols:
        df = df.drop(columns=nan_header_cols)
        meta["cleaning_notes"].append(f"Dropped {len(nan_header_cols)} column(s) with NaN headers.")

    # Normalise column names
    df.columns = [_normalise_header(c) for c in df.columns]

    # Drop fully-empty rows
    before_len = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    dropped_rows = before_len - len(df)
    if dropped_rows:
        meta["cleaning_notes"].append(f"Dropped {dropped_rows} entirely empty row(s).")

    # Coerce known inventory numeric columns
    numeric_cols = [
        "Opening Stock",
        "Purchase/Stock In",
        "Number of Units Sold",
        "Hand-In-Stock",
        "Cost Price Per Unit (USD)",
        "Cost Price Total (USD)",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Auto-coerce any other column that contains numeric strings
    for c in df.columns:
        if c not in numeric_cols and df[c].dtype == object:
            sample = df[c].dropna().head(20).astype(str)
            cleaned_sample = sample.str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
            if not cleaned_sample.empty and pd.to_numeric(cleaned_sample, errors="coerce").notna().mean() > 0.7:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip(),
                    errors="coerce"
                )

    meta["final_shape"] = df.shape
    meta["columns"] = list(df.columns)
    return df, meta


def load_inventory_excel(
    file_path_or_buffer,
    *,
    skip_title_rows: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Backwards-compatible wrapper for load_dataset.
    """
    return load_dataset(file_path_or_buffer)


def _detect_header_row(raw: pd.DataFrame) -> Optional[int]:
    """
    Detect the header row:
    1. First check if any row has 'Product ID' (DesiCrew sample layout).
    2. Check if row 0 has non-empty values across most columns (standard header).
    3. Look for the first row with multiple string values.
    """
    for idx in range(min(20, len(raw))):
        for val in raw.iloc[idx]:
            if isinstance(val, str) and "product id" in val.replace("\n", " ").strip().lower():
                return idx

    # If row 0 has >= 2 non-null string headers and >= 50% non-empty
    row0 = raw.iloc[0]
    row0_strings = [v for v in row0 if isinstance(v, str) and v.strip()]
    if len(row0_strings) >= 2 and len(row0_strings) >= (len(row0.dropna()) * 0.5):
        return 0

    # Otherwise scan first 10 rows for highest non-empty string count
    best_row = 0
    best_count = 0
    for idx in range(min(10, len(raw))):
        row = raw.iloc[idx]
        str_count = sum(1 for v in row if isinstance(v, str) and v.strip())
        if str_count > best_count:
            best_count = str_count
            best_row = idx

    return best_row if best_count >= 2 else 0


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary statistics and metadata for the loaded dataset.
    """
    summary: Dict[str, Any] = {
        "total_products": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
    }

    # Numeric summaries
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        summary["numeric_stats"] = {}
        for col in numeric_cols:
            summary["numeric_stats"][col] = {
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
                "mean": round(float(df[col].mean()), 2) if not df[col].isna().all() else None,
                "sum": float(df[col].sum()) if not df[col].isna().all() else None,
            }

    # Top 5 most expensive products by total cost
    if "Cost Price Total (USD)" in df.columns and "Product Name" in df.columns:
        top5 = (
            df.nlargest(5, "Cost Price Total (USD)")[["Product Name", "Cost Price Total (USD)"]]
            .to_dict("records")
        )
        summary["top_5_by_total_cost"] = top5

    # Total inventory value
    if "Cost Price Total (USD)" in df.columns:
        summary["total_inventory_value"] = float(df["Cost Price Total (USD)"].sum())

    # Average cost per unit
    if "Cost Price Per Unit (USD)" in df.columns:
        summary["avg_cost_per_unit"] = round(float(df["Cost Price Per Unit (USD)"].mean()), 2)

    return summary


def get_column_info(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Return a list of dicts describing each column's name, dtype, and sample values."""
    info = []
    for col in df.columns:
        sample_vals = df[col].dropna().head(3).tolist()
        info.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notna().sum()),
            "sample_values": sample_vals,
        })
    return info
