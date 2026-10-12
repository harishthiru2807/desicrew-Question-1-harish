"""
data_loader.py — Handles loading, cleaning, and validating the inventory Excel dataset.

This module provides robust parsing of the DesiCrew Inventory-Records-Sample-Data.xlsx
file, which has a non-standard layout (title rows, blank first column, multi-line headers).
It also supports arbitrary user-uploaded Excel files with standard headers.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List


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


def load_inventory_excel(
    file_path_or_buffer,
    *,
    skip_title_rows: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load an inventory Excel file and return a clean DataFrame plus metadata.

    Parameters
    ----------
    file_path_or_buffer : str or file-like
        Path to the .xlsx file or an uploaded file buffer.
    skip_title_rows : bool
        If True, attempt to detect and skip decorative title rows above the header.

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame with canonical column names and correct dtypes.
    meta : dict
        Metadata including original_shape, detected_header_row, cleaning_notes.
    """
    meta: Dict[str, Any] = {"cleaning_notes": []}

    # --- Step 1: Read raw without header ----------------------------------
    raw = pd.read_excel(file_path_or_buffer, header=None)
    meta["raw_shape"] = raw.shape

    # --- Step 2: Detect the header row ------------------------------------
    header_row = _detect_header_row(raw)
    meta["detected_header_row"] = header_row

    if header_row is None:
        # Fallback: treat the first row as header
        header_row = 0
        meta["cleaning_notes"].append(
            "Could not auto-detect header row; using first row."
        )

    # --- Step 3: Extract header and data ----------------------------------
    headers = raw.iloc[header_row].tolist()
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = headers
    df = df.reset_index(drop=True)

    # --- Step 4: Drop fully-empty columns ---------------------------------
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        meta["cleaning_notes"].append(f"Dropped {len(empty_cols)} entirely empty column(s).")

    # Drop columns whose header is NaN
    nan_header_cols = [c for c in df.columns if (isinstance(c, float) and np.isnan(c))]
    if nan_header_cols:
        df = df.drop(columns=nan_header_cols)
        meta["cleaning_notes"].append(f"Dropped {len(nan_header_cols)} column(s) with NaN headers.")

    # --- Step 5: Normalise column names -----------------------------------
    df.columns = [_normalise_header(c) for c in df.columns]

    # --- Step 6: Drop fully-empty rows ------------------------------------
    before_len = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    dropped_rows = before_len - len(df)
    if dropped_rows:
        meta["cleaning_notes"].append(f"Dropped {dropped_rows} entirely empty row(s).")

    # --- Step 7: Coerce numeric columns -----------------------------------
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

    meta["final_shape"] = df.shape
    meta["columns"] = list(df.columns)

    return df, meta


def _detect_header_row(raw: pd.DataFrame) -> Optional[int]:
    """
    Heuristic: the header row is the first row that contains 'Product ID'
    (case-insensitive, ignoring whitespace/newlines).
    """
    for idx in range(min(20, len(raw))):
        for val in raw.iloc[idx]:
            if not isinstance(val, str):
                continue
            cleaned = val.replace("\n", " ").strip().lower()
            if "product id" in cleaned:
                return idx
    return None


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
