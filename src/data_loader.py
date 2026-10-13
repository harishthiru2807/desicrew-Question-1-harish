"""
data_loader.py — Universal dataset loader for arbitrary Excel / CSV files.

Handles:
- Excel (.xlsx, .xls) with non-standard or standard headers; multi-sheet workbooks
- CSV (.csv) and TSV (.tsv) with various encodings and delimiters
- Automatic header-row detection
- Generic numeric-coercion (no hardcoded column names)
- Metadata: raw_shape, final_shape, columns, cleaning_notes, sheet info

The DesiCrew Inventory-Records-Sample-Data.xlsx file is treated as the
*reference* dataset (used for dev/testing) but nothing in this module is
specific to it.  Any compatible Excel / CSV file uploaded by the user will be
handled identically.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Header-normalisation for the DesiCrew sample workbook only
# (multi-line cell headers produced by Excel).  All other files get their
# headers used verbatim after stripping surrounding whitespace.
# ---------------------------------------------------------------------------

_DESICREW_HEADER_MAP: Dict[str, str] = {
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
    """Strip whitespace; map known DesiCrew multi-line headers to canonical names."""
    if not isinstance(raw, str):
        return str(raw)
    key = raw.strip().lower()
    return _DESICREW_HEADER_MAP.get(key, raw.strip())


# ---------------------------------------------------------------------------
# Multi-sheet inspection helpers
# ---------------------------------------------------------------------------

def inspect_excel_sheets(file_path_or_buffer) -> List[str]:
    """
    Return the list of sheet names in an Excel workbook without loading data.

    Parameters
    ----------
    file_path_or_buffer : str, Path, or file-like buffer

    Returns
    -------
    list of sheet name strings, or [] on failure.
    """
    try:
        if hasattr(file_path_or_buffer, "seek"):
            file_path_or_buffer.seek(0)
        xl = pd.ExcelFile(file_path_or_buffer)
        return xl.sheet_names
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_dataset(
    file_path_or_buffer,
    filename: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Universal dataset loader for any tabular file.

    Parameters
    ----------
    file_path_or_buffer : str, Path, or file-like buffer
    filename : str, optional — used to determine file format
    sheet_name : str, optional — Excel sheet to load; None -> first sheet

    Returns
    -------
    df   : pd.DataFrame — cleaned DataFrame
    meta : dict — metadata: raw_shape, final_shape, columns, cleaning_notes,
                  sheets (list), active_sheet (str)
    """
    meta: Dict[str, Any] = {"cleaning_notes": [], "sheets": [], "active_sheet": None}

    # Determine the filename hint
    name = filename or ""
    if not name and hasattr(file_path_or_buffer, "name"):
        name = file_path_or_buffer.name
    elif not name and isinstance(file_path_or_buffer, (str, Path)):
        name = Path(file_path_or_buffer).name
    name_lower = str(name).lower()

    is_csv = name_lower.endswith((".csv", ".tsv", ".txt"))

    if is_csv:
        df, meta = _load_csv(file_path_or_buffer, meta)
    else:
        df, meta = _load_excel(file_path_or_buffer, meta, sheet_name=sheet_name)

    # --- Generic post-processing (applies to both CSV and Excel) ---
    df = _drop_empty(df, meta)
    df = _normalise_headers(df, meta)
    df = _drop_empty_rows(df, meta)
    df = _coerce_numerics(df, meta)

    meta["final_shape"] = df.shape
    meta["columns"] = list(df.columns)
    return df, meta


# ---------------------------------------------------------------------------
# Format-specific loaders (internal)
# ---------------------------------------------------------------------------

def _load_csv(
    file_path_or_buffer,
    meta: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a CSV / TSV file with multi-encoding fallback."""
    df = None
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            if hasattr(file_path_or_buffer, "seek"):
                file_path_or_buffer.seek(0)
            df = pd.read_csv(
                file_path_or_buffer, sep=None, engine="python", encoding=enc
            )
            meta["cleaning_notes"].append(f"Loaded CSV with {enc} encoding.")
            break
        except Exception:
            continue

    if df is None:
        if hasattr(file_path_or_buffer, "seek"):
            file_path_or_buffer.seek(0)
        df = pd.read_csv(
            file_path_or_buffer, encoding="utf-8", on_bad_lines="skip"
        )
        meta["cleaning_notes"].append("Loaded CSV with utf-8 fallback (bad lines skipped).")

    meta["raw_shape"] = df.shape
    # Flatten any multi-line column names produced by merged-cell exports
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    return df, meta


def _load_excel(
    file_path_or_buffer,
    meta: Dict[str, Any],
    sheet_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load one sheet from an Excel workbook.
    Records all available sheet names in meta["sheets"].
    """
    # Discover sheet names first
    try:
        if hasattr(file_path_or_buffer, "seek"):
            file_path_or_buffer.seek(0)
        xl = pd.ExcelFile(file_path_or_buffer)
        meta["sheets"] = xl.sheet_names
    except Exception:
        meta["sheets"] = []

    # Resolve which sheet to load
    sheets = meta["sheets"]
    if sheet_name and sheet_name in sheets:
        active_sheet = sheet_name
    elif sheets:
        active_sheet = sheets[0]
    else:
        active_sheet = 0  # let pandas pick

    meta["active_sheet"] = active_sheet if isinstance(active_sheet, str) else None

    # Load the sheet without a header so we can detect the real header row
    try:
        if hasattr(file_path_or_buffer, "seek"):
            file_path_or_buffer.seek(0)
        raw = pd.read_excel(
            file_path_or_buffer,
            sheet_name=active_sheet,
            header=None,
        )
        meta["raw_shape"] = raw.shape
        header_row = _detect_header_row(raw)
        meta["detected_header_row"] = header_row

        if header_row is None:
            header_row = 0
            meta["cleaning_notes"].append("Using first row as header (auto-detected).")

        headers = raw.iloc[header_row].tolist()
        df = raw.iloc[header_row + 1:].copy()
        df.columns = headers
        df = df.reset_index(drop=True)
        meta["cleaning_notes"].append(
            f"Loaded sheet '{active_sheet}' — header at row {header_row}."
        )
    except Exception as exc:
        meta["cleaning_notes"].append(f"Header-detection failed ({exc}); using pandas defaults.")
        if hasattr(file_path_or_buffer, "seek"):
            file_path_or_buffer.seek(0)
        df = pd.read_excel(file_path_or_buffer, sheet_name=active_sheet)
        meta["raw_shape"] = df.shape

    return df, meta


# ---------------------------------------------------------------------------
# Post-processing helpers (format-agnostic)
# ---------------------------------------------------------------------------

def _drop_empty(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    """Drop columns that are entirely NaN or have a NaN/float header."""
    empty_data_cols = [c for c in df.columns if df[c].isna().all()]
    nan_header_cols = [
        c for c in df.columns if isinstance(c, float) and np.isnan(c)
    ]
    to_drop = list(set(empty_data_cols) | set(nan_header_cols))
    if to_drop:
        df = df.drop(columns=to_drop)
        meta["cleaning_notes"].append(
            f"Dropped {len(to_drop)} empty/unnamed column(s)."
        )
    return df


def _normalise_headers(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    """Apply DesiCrew multi-line header map; strip whitespace from all headers."""
    df.columns = [_normalise_header(c) for c in df.columns]
    return df


def _drop_empty_rows(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    """Drop rows that are entirely NaN."""
    before = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        meta["cleaning_notes"].append(f"Dropped {dropped} entirely empty row(s).")
    return df


def _coerce_numerics(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Auto-detect and coerce columns whose values look numeric.
    Works on any dataset — no hardcoded column names.
    Handles currency prefixes ($ pound-sign euro-sign) and thousands separators.
    """
    coerced = []
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(30).astype(str)
            cleaned = (
                sample
                .str.replace(r"[$\xa3\u20ac,\s]", "", regex=True)
                .str.strip()
            )
            if (
                not cleaned.empty
                and pd.to_numeric(cleaned, errors="coerce").notna().mean() > 0.70
            ):
                df[col] = pd.to_numeric(
                    df[col]
                    .astype(str)
                    .str.replace(r"[$\xa3\u20ac,\s]", "", regex=True)
                    .str.strip(),
                    errors="coerce",
                )
                coerced.append(col)
    if coerced:
        meta["cleaning_notes"].append(
            f"Auto-coerced {len(coerced)} column(s) to numeric: "
            f"{', '.join(coerced[:5])}{'...' if len(coerced) > 5 else ''}."
        )
    return df


# ---------------------------------------------------------------------------
# Header-row detection
# ---------------------------------------------------------------------------

def _detect_header_row(raw: pd.DataFrame) -> Optional[int]:
    """
    Heuristic header-row detection:
    1. Look for a row containing 'product id' (DesiCrew sample layout).
    2. Check if row 0 is a valid header (>=2 non-empty strings, >=50 % fill).
    3. Pick the row (among first 10) with the most non-empty string cells.
    """
    for idx in range(min(20, len(raw))):
        for val in raw.iloc[idx]:
            if (
                isinstance(val, str)
                and "product id" in val.replace("\n", " ").strip().lower()
            ):
                return idx

    row0 = raw.iloc[0]
    row0_strings = [v for v in row0 if isinstance(v, str) and v.strip()]
    if len(row0_strings) >= 2 and len(row0_strings) >= (len(row0.dropna()) * 0.5):
        return 0

    best_row, best_count = 0, 0
    for idx in range(min(10, len(raw))):
        row = raw.iloc[idx]
        str_count = sum(1 for v in row if isinstance(v, str) and v.strip())
        if str_count > best_count:
            best_count, best_row = str_count, idx

    return best_row if best_count >= 2 else 0


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers
# ---------------------------------------------------------------------------

def load_inventory_excel(
    file_path_or_buffer,
    *,
    skip_title_rows: bool = True,
    sheet_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Backwards-compatible wrapper — delegates to load_dataset."""
    return load_dataset(file_path_or_buffer, sheet_name=sheet_name)


# ---------------------------------------------------------------------------
# Dataset summary — fully generic (no hardcoded column assumptions)
# ---------------------------------------------------------------------------

def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary statistics and metadata for any loaded dataset.
    Inventory-specific keys (total_inventory_value, avg_cost_per_unit, ...)
    are only populated when the relevant columns actually exist.
    """
    summary: Dict[str, Any] = {
        "total_products": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary["numeric_columns"] = numeric_cols
    if numeric_cols:
        summary["numeric_stats"] = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            summary["numeric_stats"][col] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": round(float(series.mean()), 4),
                "sum": float(series.sum()),
            }

    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    summary["categorical_columns"] = cat_cols

    # DesiCrew inventory-specific extras (only when columns present)
    if "Cost Price Total (USD)" in df.columns and "Product Name" in df.columns:
        top5 = (
            df.nlargest(5, "Cost Price Total (USD)")[
                ["Product Name", "Cost Price Total (USD)"]
            ]
            .to_dict("records")
        )
        summary["top_5_by_total_cost"] = top5

    if "Cost Price Total (USD)" in df.columns:
        summary["total_inventory_value"] = float(
            df["Cost Price Total (USD)"].sum()
        )

    if "Cost Price Per Unit (USD)" in df.columns:
        summary["avg_cost_per_unit"] = round(
            float(df["Cost Price Per Unit (USD)"].mean()), 2
        )

    return summary


def get_column_info(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return per-column metadata: name, dtype, non-null count, sample values."""
    info = []
    for col in df.columns:
        sample_vals = df[col].dropna().head(3).tolist()
        info.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "sample_values": sample_vals,
            }
        )
    return info
