"""
tests/test_agent.py — Unit tests for agent functions, multi-provider support,
offline fallback engine, and quota error handling.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import (
    get_api_client,
    run_agent,
    run_agent_without_llm,
    PROVIDER_CONFIGS,
)


@pytest.fixture
def sample_inventory_df():
    """Create a sample inventory DataFrame for testing."""
    return pd.DataFrame({
        "Product ID": ["P101", "P102", "P103", "P104"],
        "Product Name": ["Smart Watch", "Wireless Mouse", "Gaming Keyboard", "USB-C Hub"],
        "Opening Stock": [50, 100, 80, 20],
        "Purchase/Stock In": [20, 30, 10, 15],
        "Number of Units Sold": [15, 40, 25, 5],
        "Hand-In-Stock": [55, 90, 65, 30],
        "Cost Price Per Unit (USD)": [199.99, 29.99, 89.99, 45.00],
        "Cost Price Total (USD)": [9999.50, 2999.00, 7199.20, 900.00],
    })


class TestProviderConfigs:
    def test_providers_defined(self):
        assert "Groq" in PROVIDER_CONFIGS
        assert "Gemini" in PROVIDER_CONFIGS
        assert "OpenAI" in PROVIDER_CONFIGS
        assert "Custom" in PROVIDER_CONFIGS

    def test_groq_config_properties(self):
        groq = PROVIDER_CONFIGS["Groq"]
        assert "api.groq.com" in groq["base_url"]
        assert "llama" in groq["default_model"]

    def test_gemini_config_properties(self):
        gemini = PROVIDER_CONFIGS["Gemini"]
        assert "generativelanguage.googleapis.com" in gemini["base_url"]
        assert "gemini" in gemini["default_model"]


class TestGetApiClient:
    def test_groq_key_auto_detection(self):
        with patch("openai.OpenAI") as mock_openai:
            client, model = get_api_client(api_key="gsk_12345abcdef")
            assert client is not None
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert "api.groq.com" in call_kwargs["base_url"]
            assert call_kwargs["api_key"] == "gsk_12345abcdef"

    def test_gemini_key_auto_detection(self):
        with patch("openai.OpenAI") as mock_openai:
            client, model = get_api_client(api_key="AIzaSy123456789")
            assert client is not None
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert "generativelanguage.googleapis.com" in call_kwargs["base_url"]

    def test_missing_key_returns_helpful_message(self):
        with patch("dotenv.load_dotenv"), patch.dict("os.environ", {}, clear=True):
            client, msg = get_api_client(api_key=None)
            assert client is None
            assert "API key not configured" in msg


class TestOfflineAgent:
    def test_count_query(self, sample_inventory_df):
        res = run_agent_without_llm("how many products are there?", sample_inventory_df)
        assert "4 products" in res["response"]
        assert len(res["tools_used"]) > 0

    def test_columns_query(self, sample_inventory_df):
        res = run_agent_without_llm("what columns are in this data?", sample_inventory_df)
        assert "Product ID" in res["response"]
        assert "Hand-In-Stock" in res["response"]

    def test_missing_values_query(self, sample_inventory_df):
        res = run_agent_without_llm("are there any missing values?", sample_inventory_df)
        assert "no missing values" in res["response"].lower()

    def test_total_inventory_value_query(self, sample_inventory_df):
        res = run_agent_without_llm("what is the total inventory value?", sample_inventory_df)
        assert "$21,097.70" in res["response"] or "Total Inventory Value" in res["response"]
        assert res["tools_used"][0]["success"] is True

    def test_most_expensive_product_query(self, sample_inventory_df):
        res = run_agent_without_llm("what is the most expensive product?", sample_inventory_df)
        assert "most expensive" in res["response"].lower()
        assert res["table_data"] is not None
        assert res["table_data"].iloc[0]["Product Name"] == "Smart Watch"

    def test_low_stock_query(self, sample_inventory_df):
        res = run_agent_without_llm("show me items with low stock", sample_inventory_df)
        assert "lowest stock" in res["response"].lower()
        assert res["table_data"] is not None

    def test_top_selling_query(self, sample_inventory_df):
        res = run_agent_without_llm("what are the best selling products?", sample_inventory_df)
        assert "best-selling" in res["response"].lower()
        assert res["table_data"] is not None

    def test_discrepancy_audit_query(self, sample_inventory_df):
        # In sample_inventory_df:
        # P101: 50 + 20 - 15 = 55 (matches Hand-In-Stock 55)
        # P102: 100 + 30 - 40 = 90 (matches Hand-In-Stock 90)
        # P103: 80 + 10 - 25 = 65 (matches Hand-In-Stock 65)
        # P104: 20 + 15 - 5 = 30 (matches Hand-In-Stock 30)
        # Let's add a discrepancy
        sample_inventory_df.loc[sample_inventory_df["Product ID"] == "P101", "Hand-In-Stock"] = 999
        res = run_agent_without_llm("check for inventory discrepancies or errors", sample_inventory_df)
        assert "discrepancies" in res["response"].lower()
        assert res["table_data"] is not None
        assert len(res["table_data"]) == 1

    def test_product_search_by_name(self, sample_inventory_df):
        res = run_agent_without_llm("what is the price of wireless mouse?", sample_inventory_df)
        assert "wireless mouse" in res["response"].lower()
        assert res["table_data"] is not None


class TestRunAgentQuotaFallback:
    def test_quota_error_triggers_offline_fallback(self, sample_inventory_df):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Error code: 429 - {'error': {'message': 'You have no credits remaining. "
            "Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.', "
            "'type': 'insufficient_quota', 'code': 'credit_balance_exhausted'}}"
        )

        res = run_agent(
            user_message="what is the total inventory value?",
            df=sample_inventory_df,
            chat_history=[],
            client=mock_client,
            model="gpt-4o-mini",
        )

        # Should NOT just fail with raw error, but answer with offline fallback + banner
        assert "Error 429" in res["response"]
        assert "Groq" in res["response"]
        assert "Total Inventory Value" in res["response"]
        assert res["tools_used"][0]["success"] is True
