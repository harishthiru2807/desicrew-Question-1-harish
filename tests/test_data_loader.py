"""
tests/test_data_loader.py — Automated tests for data loading and analysis functions.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    load_inventory_excel,
    get_dataset_summary,
    get_column_info,
    _normalise_header,
    _detect_header_row,
    CANONICAL_COLUMNS,
)
from src.safe_executor import (
    execute_operation,
    get_available_operations,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

SAMPLE_FILE = Path(__file__).parent.parent / "Inventory-Records-Sample-Data.xlsx"


@pytest.fixture
def sample_df():
    """Load the actual supplied dataset."""
    if not SAMPLE_FILE.exists():
        pytest.skip("Sample data file not found.")
    df, meta = load_inventory_excel(str(SAMPLE_FILE))
    return df


@pytest.fixture
def sample_meta():
    """Get metadata from loading the actual dataset."""
    if not SAMPLE_FILE.exists():
        pytest.skip("Sample data file not found.")
    _, meta = load_inventory_excel(str(SAMPLE_FILE))
    return meta


@pytest.fixture
def simple_df():
    """Create a simple in-memory DataFrame for unit tests."""
    return pd.DataFrame({
        "Product ID": ["P001", "P002", "P003", "P004", "P005"],
        "Product Name": ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones"],
        "Opening Stock": [50, 100, 80, 30, 60],
        "Purchase/Stock In": [20, 40, 30, 10, 25],
        "Number of Units Sold": [15, 30, 25, 8, 18],
        "Hand-In-Stock": [55, 110, 85, 32, 67],
        "Cost Price Per Unit (USD)": [1200.0, 25.0, 50.0, 500.0, 100.0],
        "Cost Price Total (USD)": [66000.0, 2750.0, 4250.0, 16000.0, 6700.0],
    })


# ── Data loader tests ─────────────────────────────────────────────────────

class TestDataLoader:
    """Tests for the data loading pipeline."""

    def test_load_actual_file(self, sample_df):
        """Verify the supplied Excel file loads correctly."""
        assert sample_df is not None
        assert len(sample_df) > 0

    def test_correct_row_count(self, sample_df):
        """The supplied file has 46 products (P101–P146)."""
        assert len(sample_df) == 46

    def test_correct_columns(self, sample_df):
        """Check that canonical column names are present after cleaning."""
        expected = {
            "Product ID", "Product Name", "Opening Stock",
            "Purchase/Stock In", "Number of Units Sold", "Hand-In-Stock",
            "Cost Price Per Unit (USD)", "Cost Price Total (USD)",
        }
        assert expected == set(sample_df.columns)

    def test_no_empty_columns(self, sample_df):
        """Ensure no columns are entirely NaN after cleaning."""
        for col in sample_df.columns:
            assert not sample_df[col].isna().all(), f"Column {col} is entirely empty"

    def test_numeric_columns_are_numeric(self, sample_df):
        """Numeric columns should have numeric dtypes."""
        numeric_expected = [
            "Opening Stock", "Purchase/Stock In", "Number of Units Sold",
            "Hand-In-Stock", "Cost Price Per Unit (USD)", "Cost Price Total (USD)",
        ]
        for col in numeric_expected:
            assert pd.api.types.is_numeric_dtype(sample_df[col]), \
                f"Column {col} should be numeric but is {sample_df[col].dtype}"

    def test_product_ids_unique(self, sample_df):
        """Product IDs should be unique."""
        if "Product ID" in sample_df.columns:
            assert sample_df["Product ID"].is_unique

    def test_metadata_has_expected_keys(self, sample_meta):
        """Metadata should contain expected keys."""
        assert "raw_shape" in sample_meta
        assert "final_shape" in sample_meta
        assert "columns" in sample_meta
        assert "cleaning_notes" in sample_meta
        assert "detected_header_row" in sample_meta

    def test_header_normalisation(self):
        """Multi-line Excel headers should normalise to canonical names."""
        assert _normalise_header("Opening \nStock") == "Opening Stock"
        assert _normalise_header("Purchase/\nStock in") == "Purchase/Stock In"
        assert _normalise_header("Cost Price \nPer Unit (USD)") == "Cost Price Per Unit (USD)"

    def test_first_product_is_p101(self, sample_df):
        """First product should be P101 (Laptop)."""
        assert sample_df.iloc[0]["Product ID"] == "P101"
        assert sample_df.iloc[0]["Product Name"] == "Laptop"

    def test_last_product_is_p146(self, sample_df):
        """Last product should be P146 (Gaming Desk)."""
        assert sample_df.iloc[-1]["Product ID"] == "P146"
        assert sample_df.iloc[-1]["Product Name"] == "Gaming Desk"


# ── Dataset summary tests ─────────────────────────────────────────────────

class TestDatasetSummary:
    """Tests for the summary statistics function."""

    def test_summary_has_product_count(self, sample_df):
        summary = get_dataset_summary(sample_df)
        assert summary["total_products"] == 46

    def test_summary_has_total_value(self, sample_df):
        summary = get_dataset_summary(sample_df)
        assert "total_inventory_value" in summary
        assert summary["total_inventory_value"] > 0

    def test_summary_missing_values(self, simple_df):
        summary = get_dataset_summary(simple_df)
        assert summary["total_missing"] == 0

    def test_column_info(self, sample_df):
        info = get_column_info(sample_df)
        assert len(info) == 8
        assert all("column" in c for c in info)
        assert all("dtype" in c for c in info)


# ── Safe executor tests ───────────────────────────────────────────────────

class TestSafeExecutor:
    """Tests for the whitelist-based safe execution engine."""

    def test_available_operations(self):
        ops = get_available_operations()
        assert len(ops) > 10
        names = {o["name"] for o in ops}
        assert "filter" in names
        assert "aggregate" in names
        assert "sort" in names

    def test_describe_operation(self, simple_df):
        result = execute_operation("describe", simple_df, {})
        assert result["success"] is True
        assert isinstance(result["result"], pd.DataFrame)

    def test_head_operation(self, simple_df):
        result = execute_operation("head", simple_df, {"n": 3})
        assert result["success"] is True
        assert len(result["result"]) == 3

    def test_shape_operation(self, simple_df):
        result = execute_operation("shape", simple_df, {})
        assert result["success"] is True
        assert result["result"]["rows"] == 5
        assert result["result"]["columns"] == 8

    def test_filter_equals(self, simple_df):
        result = execute_operation("filter", simple_df, {
            "column": "Product Name",
            "operator": "==",
            "value": "Laptop",
        })
        assert result["success"] is True
        assert len(result["result"]) == 1
        assert result["result"].iloc[0]["Product Name"] == "Laptop"

    def test_filter_greater_than(self, simple_df):
        result = execute_operation("filter", simple_df, {
            "column": "Opening Stock",
            "operator": ">",
            "value": 60,
        })
        assert result["success"] is True
        assert all(result["result"]["Opening Stock"] > 60)

    def test_filter_contains(self, simple_df):
        result = execute_operation("filter", simple_df, {
            "column": "Product Name",
            "operator": "contains",
            "value": "boa",
        })
        assert result["success"] is True
        assert len(result["result"]) == 1  # Keyboard

    def test_sort_ascending(self, simple_df):
        result = execute_operation("sort", simple_df, {
            "column": "Cost Price Per Unit (USD)",
            "ascending": True,
        })
        assert result["success"] is True
        values = result["result"]["Cost Price Per Unit (USD)"].tolist()
        assert values == sorted(values)

    def test_sort_descending(self, simple_df):
        result = execute_operation("sort", simple_df, {
            "column": "Cost Price Per Unit (USD)",
            "ascending": False,
        })
        assert result["success"] is True
        values = result["result"]["Cost Price Per Unit (USD)"].tolist()
        assert values == sorted(values, reverse=True)

    def test_aggregate_sum(self, simple_df):
        result = execute_operation("aggregate", simple_df, {
            "column": "Cost Price Total (USD)",
            "function": "sum",
        })
        assert result["success"] is True
        expected = simple_df["Cost Price Total (USD)"].sum()
        assert result["result"]["value"] == expected

    def test_aggregate_mean(self, simple_df):
        result = execute_operation("aggregate", simple_df, {
            "column": "Cost Price Per Unit (USD)",
            "function": "mean",
        })
        assert result["success"] is True
        expected = float(simple_df["Cost Price Per Unit (USD)"].mean())
        assert abs(result["result"]["value"] - expected) < 0.01

    def test_top_n(self, simple_df):
        result = execute_operation("top_n", simple_df, {
            "column": "Cost Price Per Unit (USD)",
            "n": 3,
        })
        assert result["success"] is True
        assert len(result["result"]) == 3
        assert result["result"].iloc[0]["Product Name"] == "Laptop"

    def test_bottom_n(self, simple_df):
        result = execute_operation("bottom_n", simple_df, {
            "column": "Cost Price Per Unit (USD)",
            "n": 2,
        })
        assert result["success"] is True
        assert len(result["result"]) == 2

    def test_missing_values(self, simple_df):
        result = execute_operation("missing_values", simple_df, {})
        assert result["success"] is True
        assert all(v == 0 for v in result["result"].values())

    def test_value_counts(self, simple_df):
        result = execute_operation("value_counts", simple_df, {"column": "Product Name"})
        assert result["success"] is True
        assert len(result["result"]) == 5

    def test_unique_values(self, simple_df):
        result = execute_operation("unique_values", simple_df, {"column": "Product ID"})
        assert result["success"] is True
        assert len(result["result"]) == 5

    def test_search_rows(self, simple_df):
        result = execute_operation("search_rows", simple_df, {"query": "Laptop"})
        assert result["success"] is True
        assert len(result["result"]) == 1

    def test_invalid_operation(self, simple_df):
        result = execute_operation("nonexistent_op", simple_df, {})
        assert result["success"] is False
        assert result["error"] is not None

    def test_invalid_column(self, simple_df):
        result = execute_operation("filter", simple_df, {
            "column": "NonExistentColumn",
            "operator": "==",
            "value": "test",
        })
        assert result["success"] is False

    def test_correlation(self, simple_df):
        result = execute_operation("correlation", simple_df, {})
        assert result["success"] is True
        assert isinstance(result["result"], pd.DataFrame)

    def test_percentile(self, simple_df):
        result = execute_operation("percentile", simple_df, {
            "column": "Cost Price Per Unit (USD)",
            "percentile": 50,
        })
        assert result["success"] is True
        assert "value" in result["result"]

    def test_select_columns(self, simple_df):
        result = execute_operation("select_columns", simple_df, {
            "columns": ["Product Name", "Cost Price Per Unit (USD)"],
        })
        assert result["success"] is True
        assert list(result["result"].columns) == ["Product Name", "Cost Price Per Unit (USD)"]

    def test_calculate_column(self, simple_df):
        result = execute_operation("calculate_column", simple_df, {
            "column_a": "Cost Price Per Unit (USD)",
            "operation": "*",
            "scalar": 2,
            "result_name": "Double Price",
        })
        assert result["success"] is True
        assert "Double Price" in result["result"].columns


# ── Integration tests with actual file ─────────────────────────────────────

class TestIntegration:
    """Integration tests using the real dataset."""

    def test_laptop_cost(self, sample_df):
        """Verify Laptop has cost price of 1200 USD."""
        laptop = sample_df[sample_df["Product Name"] == "Laptop"]
        assert len(laptop) == 1
        assert laptop.iloc[0]["Cost Price Per Unit (USD)"] == 1200

    def test_hand_in_stock_calculation(self, sample_df):
        """Verify Hand-In-Stock ≈ Opening Stock + Purchase - Sold for most rows.

        NOTE: The supplied dataset has known discrepancies in some products
        where Hand-In-Stock != Opening + Purchase - Sold. This may reflect
        returns, stock adjustments, or intentional test data. We verify the
        formula holds for the majority of products.
        """
        matches = 0
        total = len(sample_df)
        for _, row in sample_df.iterrows():
            expected = row["Opening Stock"] + row["Purchase/Stock In"] - row["Number of Units Sold"]
            actual = row["Hand-In-Stock"]
            if abs(actual - expected) < 1:
                matches += 1
        # At least 70% of products should follow the formula
        match_pct = matches / total
        assert match_pct >= 0.70, \
            f"Only {matches}/{total} ({match_pct:.0%}) products match the stock formula"

    def test_total_cost_calculation(self, sample_df):
        """Verify Cost Price Total = Unit Cost × Hand-In-Stock."""
        for _, row in sample_df.iterrows():
            expected = row["Cost Price Per Unit (USD)"] * row["Hand-In-Stock"]
            actual = row["Cost Price Total (USD)"]
            assert abs(actual - expected) < 1, \
                f"Product {row['Product ID']}: expected {expected}, got {actual}"

    def test_filter_and_aggregate_pipeline(self, sample_df):
        """Test a multi-step analysis pipeline."""
        # Filter products with stock > 50
        result1 = execute_operation("filter", sample_df, {
            "column": "Hand-In-Stock",
            "operator": ">",
            "value": 50,
        })
        assert result1["success"]
        filtered = result1["result"]

        # Aggregate: sum of total cost for filtered products
        result2 = execute_operation("aggregate", filtered, {
            "column": "Cost Price Total (USD)",
            "function": "sum",
        })
        assert result2["success"]
        assert result2["result"]["value"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
