"""
safe_executor.py — Sandboxed execution of data-analysis operations.

Instead of running arbitrary eval/exec on LLM-generated code, this module provides
a whitelist of safe, parameterised operations that the agent can call. Each operation
is a pure function that takes a DataFrame and parameters, returning results.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import re


# ---------------------------------------------------------------------------
# Registry of safe operations
# ---------------------------------------------------------------------------

OPERATIONS = {}


def register(name: str):
    """Decorator to register a safe operation."""
    def decorator(fn):
        OPERATIONS[name] = fn
        return fn
    return decorator


def execute_operation(
    op_name: str,
    df: pd.DataFrame,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a named operation safely.

    Returns
    -------
    dict with keys:
        success : bool
        result  : Any   — the computed result (DataFrame, scalar, dict, etc.)
        description : str — human-readable description of what was done
        error   : str | None
    """
    if op_name not in OPERATIONS:
        return {
            "success": False,
            "result": None,
            "description": f"Unknown operation: {op_name}",
            "error": f"Operation '{op_name}' is not registered.",
        }
    try:
        result, description = OPERATIONS[op_name](df, params)
        return {
            "success": True,
            "result": result,
            "description": description,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "description": f"Error executing {op_name}",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Safe operations
# ---------------------------------------------------------------------------

@register("describe")
def op_describe(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Full descriptive statistics."""
    cols = params.get("columns")
    if cols:
        valid = [c for c in cols if c in df.columns]
        result = df[valid].describe(include="all")
    else:
        result = df.describe(include="all")
    return result, "Generated descriptive statistics for the dataset."


@register("head")
def op_head(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Return first N rows."""
    n = min(int(params.get("n", 5)), 100)
    return df.head(n), f"Showing the first {n} rows of the dataset."


@register("tail")
def op_tail(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Return last N rows."""
    n = min(int(params.get("n", 5)), 100)
    return df.tail(n), f"Showing the last {n} rows of the dataset."


@register("shape")
def op_shape(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Return shape of the dataset."""
    rows, cols = df.shape
    return {"rows": rows, "columns": cols}, f"The dataset has {rows} rows and {cols} columns."


@register("columns")
def op_columns(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """List all column names."""
    return list(df.columns), f"The dataset has {len(df.columns)} columns."


@register("dtypes")
def op_dtypes(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Data types of all columns."""
    result = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return result, "Data types for each column."


@register("missing_values")
def op_missing(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Count of missing values per column."""
    missing = df.isnull().sum()
    total = int(missing.sum())
    result = missing.to_dict()
    return result, f"Missing value analysis: {total} total missing values found."


@register("filter")
def op_filter(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """
    Filter rows by column conditions.
    params: column, operator (==, !=, >, <, >=, <=, contains, startswith, endswith), value
    """
    col = params.get("column")
    op = params.get("operator", "==")
    value = params.get("value")

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # Type coercion
    if df[col].dtype in [np.float64, np.int64, float, int]:
        try:
            value = float(value)
        except (ValueError, TypeError):
            pass

    if op == "==":
        mask = df[col] == value
    elif op == "!=":
        mask = df[col] != value
    elif op == ">":
        mask = df[col] > value
    elif op == "<":
        mask = df[col] < value
    elif op == ">=":
        mask = df[col] >= value
    elif op == "<=":
        mask = df[col] <= value
    elif op == "contains":
        mask = df[col].astype(str).str.contains(str(value), case=False, na=False)
    elif op == "startswith":
        mask = df[col].astype(str).str.startswith(str(value), na=False)
    elif op == "endswith":
        mask = df[col].astype(str).str.endswith(str(value), na=False)
    else:
        raise ValueError(f"Unsupported operator: {op}")

    result = df[mask]
    return result, f"Filtered {col} {op} {value}: {len(result)} rows matched."


@register("sort")
def op_sort(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Sort by one or more columns."""
    columns = params.get("columns", params.get("column"))
    ascending = params.get("ascending", True)

    if isinstance(columns, str):
        columns = [columns]
    valid = [c for c in columns if c in df.columns]
    if not valid:
        raise ValueError(f"None of the specified columns found. Available: {list(df.columns)}")

    result = df.sort_values(by=valid, ascending=ascending)
    direction = "ascending" if ascending else "descending"
    return result, f"Sorted by {', '.join(valid)} ({direction})."


@register("top_n")
def op_top_n(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Get top N rows by a column."""
    col = params.get("column")
    n = min(int(params.get("n", 5)), 100)

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")

    result = df.nlargest(n, col)
    return result, f"Top {n} rows by {col}."


@register("bottom_n")
def op_bottom_n(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Get bottom N rows by a column."""
    col = params.get("column")
    n = min(int(params.get("n", 5)), 100)

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")

    result = df.nsmallest(n, col)
    return result, f"Bottom {n} rows by {col}."


@register("aggregate")
def op_aggregate(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """
    Aggregate a column with a function.
    params: column, function (sum, mean, median, min, max, count, std, var, nunique)
    """
    col = params.get("column")
    func = params.get("function", "sum")

    if col and col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")

    allowed_funcs = {"sum", "mean", "median", "min", "max", "count", "std", "var", "nunique"}
    if func not in allowed_funcs:
        raise ValueError(f"Function must be one of: {allowed_funcs}")

    if col:
        series = df[col]
        if func == "nunique":
            value = int(series.nunique())
        else:
            value = float(getattr(series, func)())
        return {"column": col, "function": func, "value": value}, \
            f"The {func} of {col} is {value}."
    else:
        # Aggregate all numeric columns
        numeric = df.select_dtypes(include=[np.number])
        if func == "nunique":
            result = numeric.nunique().to_dict()
        else:
            result = getattr(numeric, func)().to_dict()
        return result, f"Applied {func} across all numeric columns."


@register("group_by")
def op_group_by(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """
    Group by one or more columns and aggregate.
    params: group_columns (list), agg_column, agg_function
    """
    group_cols = params.get("group_columns", [])
    agg_col = params.get("agg_column")
    agg_func = params.get("agg_function", "sum")

    if isinstance(group_cols, str):
        group_cols = [group_cols]

    valid_groups = [c for c in group_cols if c in df.columns]
    if not valid_groups:
        raise ValueError(f"No valid group columns. Available: {list(df.columns)}")

    allowed_funcs = {"sum", "mean", "median", "min", "max", "count", "std", "nunique"}
    if agg_func not in allowed_funcs:
        raise ValueError(f"Agg function must be one of: {allowed_funcs}")

    if agg_col and agg_col in df.columns:
        result = df.groupby(valid_groups)[agg_col].agg(agg_func).reset_index()
        result.columns = valid_groups + [f"{agg_func}_{agg_col}"]
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        valid_agg = [c for c in numeric if c not in valid_groups]
        result = df.groupby(valid_groups)[valid_agg].agg(agg_func).reset_index()

    return result, f"Grouped by {', '.join(valid_groups)} with {agg_func} aggregation."


@register("value_counts")
def op_value_counts(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Count unique values in a column."""
    col = params.get("column")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")

    n = min(int(params.get("n", 20)), 100)
    result = df[col].value_counts().head(n).to_dict()
    return result, f"Value counts for {col} (top {n})."


@register("unique_values")
def op_unique(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """List unique values of a column."""
    col = params.get("column")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")

    values = df[col].dropna().unique().tolist()
    return values, f"Found {len(values)} unique values in {col}."


@register("correlation")
def op_correlation(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Correlation matrix of numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("No numeric columns for correlation.")
    result = numeric.corr()
    return result, "Correlation matrix of numeric columns."


@register("select_columns")
def op_select(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Select specific columns."""
    cols = params.get("columns", [])
    if isinstance(cols, str):
        cols = [cols]
    valid = [c for c in cols if c in df.columns]
    if not valid:
        raise ValueError(f"No valid columns specified. Available: {list(df.columns)}")
    return df[valid], f"Selected columns: {', '.join(valid)}."


@register("search_rows")
def op_search(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Search for a text pattern across all string columns."""
    query = str(params.get("query", ""))
    if not query:
        raise ValueError("Search query is required.")

    mask = pd.Series([False] * len(df), index=df.index)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        mask |= df[col].astype(str).str.contains(query, case=False, na=False)

    result = df[mask]
    return result, f"Search for '{query}': {len(result)} rows matched."


@register("calculate_column")
def op_calculate(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """
    Perform a safe arithmetic operation between two numeric columns or a column and a scalar.
    params: column_a, operation (+, -, *, /), column_b OR scalar, result_name
    """
    col_a = params.get("column_a")
    operation = params.get("operation", "+")
    col_b = params.get("column_b")
    scalar = params.get("scalar")
    result_name = params.get("result_name", "calculated")

    if col_a not in df.columns:
        raise ValueError(f"Column '{col_a}' not found.")

    allowed_ops = {"+", "-", "*", "/"}
    if operation not in allowed_ops:
        raise ValueError(f"Operation must be one of: {allowed_ops}")

    a = df[col_a]

    if col_b and col_b in df.columns:
        b = df[col_b]
        desc = f"{col_a} {operation} {col_b}"
    elif scalar is not None:
        b = float(scalar)
        desc = f"{col_a} {operation} {scalar}"
    else:
        raise ValueError("Must provide either column_b or scalar.")

    if operation == "+":
        result_series = a + b
    elif operation == "-":
        result_series = a - b
    elif operation == "*":
        result_series = a * b
    elif operation == "/":
        result_series = a / b.replace(0, np.nan) if isinstance(b, pd.Series) else a / (b if b != 0 else np.nan)

    result_df = df.copy()
    result_df[result_name] = result_series
    return result_df, f"Calculated {result_name} = {desc}."


@register("percentile")
def op_percentile(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Calculate a percentile of a column."""
    col = params.get("column")
    q = float(params.get("percentile", 50))

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")

    value = float(np.nanpercentile(df[col].dropna().values, q))
    return {"column": col, "percentile": q, "value": value}, \
        f"The {q}th percentile of {col} is {value}."


@register("crosstab")
def op_crosstab(df: pd.DataFrame, params: Dict) -> Tuple[Any, str]:
    """Cross-tabulation between two columns."""
    col_a = params.get("column_a")
    col_b = params.get("column_b")
    if col_a not in df.columns or col_b not in df.columns:
        raise ValueError("Both columns must exist in the dataset.")
    result = pd.crosstab(df[col_a], df[col_b])
    return result, f"Cross-tabulation of {col_a} × {col_b}."


def get_available_operations() -> List[Dict[str, str]]:
    """Return a list of all available safe operations with docstrings."""
    ops = []
    for name, fn in OPERATIONS.items():
        ops.append({
            "name": name,
            "description": (fn.__doc__ or "").strip().split("\n")[0],
        })
    return ops
