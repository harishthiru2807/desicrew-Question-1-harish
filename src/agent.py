"""
agent.py — The AI agent that orchestrates natural-language understanding,
tool selection, safe execution, and response generation.

Supports multiple LLM providers:
  - Groq (Free & Fast — recommended, no credit card required)
  - Google Gemini (Free tier via OpenAI-compatible endpoint)
  - OpenAI (gpt-4o-mini, gpt-4o)
  - Custom / Local (Ollama, LM Studio, etc.)

Also provides an extensive offline fallback engine that handles complex
queries locally without requiring an active API key or credits.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from src.safe_executor import execute_operation, get_available_operations
from src.search_tool import web_search, format_search_results
from src.data_loader import get_dataset_summary, get_column_info


# ---------------------------------------------------------------------------
# Default Gemini Configuration
# ---------------------------------------------------------------------------

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

PROVIDER_CONFIGS = {
    "Groq": {
        "display_name": "Groq (Free & Fast — Recommended)",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "env_key": "GROQ_API_KEY",
        "signup_url": "https://console.groq.com/keys",
        "badge": "100% Free Forever (No Card Required)",
    },
    "Gemini": {
        "display_name": "Google Gemini (Free Tier)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-3.8-flash",
        "models": ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.5-pro"],
        "env_key": "GEMINI_API_KEY",
        "signup_url": "https://aistudio.google.com/apikey",
        "badge": "Free Tier (Google AI Studio)",
    },
    "OpenAI": {
        "display_name": "OpenAI (Paid)",
        "base_url": None,
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o"],
        "env_key": "OPENAI_API_KEY",
        "signup_url": "https://platform.openai.com/api-keys",
        "badge": "Paid (Requires OpenAI Credits)",
    },
    "Custom": {
        "display_name": "Custom / Ollama / Local",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3",
        "models": ["llama3", "mistral"],
        "env_key": "CUSTOM_API_KEY",
        "signup_url": None,
        "badge": "Self-hosted / Local",
    },
}


# ---------------------------------------------------------------------------
# Tool definitions for function-calling
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_dataset",
            "description": (
                "Execute a safe, parameterised data operation on the inventory dataset. "
                "Available operations: describe, head, tail, shape, columns, dtypes, "
                "missing_values, filter, sort, top_n, bottom_n, aggregate, group_by, "
                "value_counts, unique_values, correlation, select_columns, search_rows, "
                "calculate_column, percentile, crosstab. "
                "Use this tool to answer questions that can be answered from the dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The name of the operation to execute.",
                        "enum": [
                            "describe", "head", "tail", "shape", "columns", "dtypes",
                            "missing_values", "filter", "sort", "top_n", "bottom_n",
                            "aggregate", "group_by", "value_counts", "unique_values",
                            "correlation", "select_columns", "search_rows",
                            "calculate_column", "percentile", "crosstab",
                        ],
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            "Parameters for the operation. Common params: "
                            "column (str), columns (list), n (int), operator (str), "
                            "value (any), ascending (bool), function (str: sum/mean/median/min/max/count/std/var/nunique), "
                            "group_columns (list), agg_column (str), agg_function (str), "
                            "query (str for search_rows), column_a (str), column_b (str), "
                            "scalar (number), operation (+/-/*/), result_name (str), "
                            "percentile (number 0-100)."
                        ),
                    },
                },
                "required": ["operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for definitions, explanations, or contextual information "
                "that cannot be found in the dataset. Use this when the user asks about "
                "industry terms, acronyms, best practices, or anything not in the data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def _get_system_prompt(df: pd.DataFrame) -> str:
    """Build the system prompt with dataset context."""
    summary = get_dataset_summary(df)
    col_info = get_column_info(df)

    col_descriptions = "\n".join(
        f"  - {c['column']} ({c['dtype']}): e.g. {c['sample_values'][:3]}"
        for c in col_info
    )

    return f"""You are an AI Data Analysis Agent for the DesiCrew Solutions Data Science Internship.
You help users analyze an inventory dataset by answering questions in plain English.

## Dataset Overview
- Total products: {summary['total_products']}
- Columns: {', '.join(summary['columns'])}
- Total inventory value: ${summary.get('total_inventory_value', 'N/A'):,.2f}

## Column Details
{col_descriptions}

## Your Capabilities
1. **query_dataset**: Execute safe data operations (filter, sort, aggregate, group, search, etc.)
2. **web_search**: Look up definitions, industry terms, or external context

## Rules
- Always use the query_dataset tool to answer data questions — never guess or make up numbers.
- Use web_search only for definitions or context not available in the dataset.
- When showing results, be clear whether the answer comes from the dataset or from external search.
- Present numeric results with appropriate formatting (commas, dollar signs, etc.).
- If a question cannot be answered with the available tools, say so honestly.
- Suggest visualizations (charts/tables) when they would help communicate the answer.
- If the user's question is ambiguous, ask for clarification.
- For complex questions, you may call multiple tools sequentially.
- Always summarize findings in clear, professional English.
- When you use a tool, briefly mention which tool and operation you used.

## Response Format
- Start with a brief direct answer.
- Show supporting data (tables, numbers) if relevant.
- End with any additional insights or suggestions if appropriate.
- Mark data-sourced answers vs. external information clearly.
"""


def get_api_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[Optional[Any], str]:
    """
    Create an OpenAI-compatible client using environment variables, secrets,
    or explicit parameters. Supports Groq, Google Gemini, OpenAI, and custom endpoints.

    Returns (client, model_name) or (None, error_message).
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None, "OpenAI package not installed. Run: pip install openai"

    # Try loading .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Streamlit secrets helper
    st_secrets = {}
    try:
        import streamlit as st
        st_secrets = dict(st.secrets)
    except Exception:
        pass

    # Check environment / secrets if not explicitly supplied
    if not api_key:
        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or st_secrets.get("GEMINI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or st_secrets.get("OPENAI_API_KEY")
            or os.environ.get("GROQ_API_KEY")
            or st_secrets.get("GROQ_API_KEY")
        )

    if not api_key:
        return None, "API key not configured."

    # Resolve provider
    if not provider:
        if api_key.startswith("gsk_"):
            provider = "Groq"
        elif api_key.startswith("AIza") or api_key.startswith("AQ."):
            provider = "Gemini"
        elif api_key.startswith("sk-"):
            provider = "OpenAI"
        else:
            provider = "Gemini"

    cfg = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["Gemini"])
    effective_base_url = base_url or cfg["base_url"]
    effective_model = model or cfg["default_model"]

    try:
        client_kwargs = {"api_key": api_key}
        if effective_base_url:
            client_kwargs["base_url"] = effective_base_url
        client = OpenAI(**client_kwargs)
        # Store metadata on client
        client._provider_name = provider
        client._model_name = effective_model
        return client, effective_model
    except Exception as e:
        return None, f"Failed to initialize client: {str(e)}"


def run_agent(
    user_message: str,
    df: pd.DataFrame,
    chat_history: List[Dict[str, str]],
    client=None,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Process a user message through the AI agent.

    Returns
    -------
    dict with keys:
        response      : str     — the final text response
        tools_used    : list    — list of tool invocations and results
        chart_data    : dict | None — data for chart rendering
        table_data    : pd.DataFrame | None — data for table rendering
        error         : str | None
    """
    result = {
        "response": "",
        "tools_used": [],
        "chart_data": None,
        "table_data": None,
        "error": None,
    }

    if client is None:
        result["error"] = "API client not available."
        result["response"] = "I'm unable to process your request because the AI API is not configured."
        return result

    # Build messages
    system_prompt = _get_system_prompt(df)
    messages = [{"role": "system", "content": system_prompt}]

    # Add recent chat history (last 20 messages for context)
    for msg in chat_history[-20:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        # Initial call with tools
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=4096,
        )

        message = response.choices[0].message

        # Process tool calls (up to 5 iterations to prevent infinite loops)
        max_iterations = 5
        iteration = 0

        while message.tool_calls and iteration < max_iterations:
            iteration += 1
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                tool_result = _execute_tool(tool_name, tool_args, df)
                result["tools_used"].append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result_summary": tool_result.get("description", ""),
                    "success": tool_result.get("success", False),
                })

                # Capture table/chart data from tool results
                if tool_result.get("success") and isinstance(tool_result.get("result"), pd.DataFrame):
                    result["table_data"] = tool_result["result"]
                elif tool_result.get("success") and isinstance(tool_result.get("result"), dict):
                    if _is_chartable(tool_result["result"]):
                        result["chart_data"] = tool_result["result"]

                # Serialise tool result for the LLM
                serialised = _serialise_tool_result(tool_result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": serialised,
                })

            # Get next response
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=4096,
            )
            message = response.choices[0].message

        result["response"] = message.content or "I completed the analysis but have no additional comments."

    except Exception as e:
        err_msg = str(e)
        result["error"] = err_msg

        # Seamlessly fall back to offline query execution
        fallback_res = run_agent_without_llm(user_message, df)
        if fallback_res.get("response"):
            result["response"] = fallback_res["response"]
            result["tools_used"] = fallback_res.get("tools_used", [])
            result["table_data"] = fallback_res.get("table_data")
            result["chart_data"] = fallback_res.get("chart_data")
        else:
            result["response"] = f"I encountered an issue processing your request: {err_msg}"

    return result


def _execute_tool(
    tool_name: str,
    args: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Dispatch a tool call to the appropriate handler."""
    if tool_name == "query_dataset":
        operation = args.get("operation", "")
        params = args.get("params", {})
        return execute_operation(operation, df, params)

    elif tool_name == "web_search":
        query = args.get("query", "")
        search_result = web_search(query)
        formatted = format_search_results(search_result)
        return {
            "success": search_result["success"],
            "result": formatted,
            "description": f"Web search for: {query}",
            "error": search_result.get("error"),
        }

    else:
        return {
            "success": False,
            "result": None,
            "description": f"Unknown tool: {tool_name}",
            "error": f"Tool '{tool_name}' is not available.",
        }


def _serialise_tool_result(tool_result: Dict[str, Any]) -> str:
    """Convert a tool result to a JSON string for the LLM context."""
    result = tool_result.get("result")

    if isinstance(result, pd.DataFrame):
        if len(result) > 50:
            serialised_data = result.head(50).to_string(index=False)
            serialised_data += f"\n... ({len(result)} total rows, showing first 50)"
        else:
            serialised_data = result.to_string(index=False)
        return json.dumps({
            "success": tool_result["success"],
            "description": tool_result["description"],
            "data": serialised_data,
            "row_count": len(result),
            "columns": list(result.columns),
        })

    elif isinstance(result, (dict, list)):
        return json.dumps({
            "success": tool_result["success"],
            "description": tool_result["description"],
            "data": result,
        }, default=str)

    else:
        return json.dumps({
            "success": tool_result["success"],
            "description": tool_result["description"],
            "data": str(result),
            "error": tool_result.get("error"),
        }, default=str)


def _is_chartable(data: Dict) -> bool:
    """Heuristic: is this dict suitable for a simple chart?"""
    if not isinstance(data, dict):
        return False
    if len(data) >= 2:
        values = list(data.values())
        return all(isinstance(v, (int, float)) for v in values)
    return False


def run_agent_without_llm(
    user_message: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    A comprehensive fallback agent that answers queries without an LLM.
    Uses pattern matching and direct data operations to answer common
    inventory questions reliably.
    """
    result = {
        "response": "",
        "tools_used": [],
        "chart_data": None,
        "table_data": None,
        "error": None,
    }

    msg = user_message.lower().strip()

    # 1. Row count / Total products
    if any(w in msg for w in ["how many products", "total products", "number of products", "product count", "how many items", "row count"]):
        result["response"] = f"The dataset contains **{len(df)} products** across **{len(df.columns)} columns**."
        result["tools_used"].append({
            "tool": "query_dataset", "args": {"operation": "shape"},
            "result_summary": f"{len(df)} rows, {len(df.columns)} cols", "success": True,
        })

    # 2. Columns / Schema
    elif any(w in msg for w in ["columns", "fields", "attributes", "what data", "schema", "features"]):
        cols = ", ".join(df.columns)
        result["response"] = f"The dataset has **{len(df.columns)} columns**:\n\n" + "\n".join(f"- `{c}`" for c in df.columns)
        result["tools_used"].append({
            "tool": "query_dataset", "args": {"operation": "columns"},
            "result_summary": f"{len(df.columns)} columns", "success": True,
        })

    # 3. Missing values
    elif any(w in msg for w in ["missing", "null", "empty", "nan", "blank"]):
        missing = df.isnull().sum()
        total = int(missing.sum())
        if total == 0:
            result["response"] = "✅ There are **no missing values** anywhere in the dataset! All fields are complete."
        else:
            details = "\n".join(f"- **{col}**: {count} missing" for col, count in missing.items() if count > 0)
            result["response"] = f"Found **{total} total missing values** across the dataset:\n{details}"
        result["tools_used"].append({
            "tool": "query_dataset", "args": {"operation": "missing_values"},
            "result_summary": f"{total} missing values", "success": True,
        })

    # 4. Total inventory valuation
    elif any(w in msg for w in ["total value", "total cost", "inventory value", "worth", "valuation"]):
        if "Cost Price Total (USD)" in df.columns:
            total = df["Cost Price Total (USD)"].sum()
            avg = df["Cost Price Total (USD)"].mean()
            result["response"] = (
                f"💰 **Total Inventory Value:** **${total:,.2f}**\n\n"
                f"- **Average product total cost:** ${avg:,.2f}\n"
                f"- **Calculated across:** {len(df)} inventory items"
            )
            result["tools_used"].append({
                "tool": "query_dataset", "args": {"operation": "aggregate", "function": "sum", "column": "Cost Price Total (USD)"},
                "result_summary": f"Total: ${total:,.2f}", "success": True,
            })
        else:
            result["response"] = "Cost price total column not found in the dataset."

    # 5. Most expensive products
    elif any(w in msg for w in ["most expensive", "highest cost", "top price", "costliest", "highest unit price", "priciest"]):
        col = "Cost Price Per Unit (USD)" if "Cost Price Per Unit (USD)" in df.columns else "Cost Price Total (USD)"
        if col in df.columns:
            display_cols = [c for c in ["Product ID", "Product Name", col] if c in df.columns]
            top = df.nlargest(5, col)[display_cols]
            result["response"] = f"**Top 5 most expensive products by `{col}`:**"
            result["table_data"] = top
            result["tools_used"].append({
                "tool": "query_dataset", "args": {"operation": "top_n", "column": col, "n": 5},
                "result_summary": f"Top 5 by {col}", "success": True,
            })
        else:
            result["response"] = "Price column not found in dataset."

    # 6. Cheapest products
    elif any(w in msg for w in ["cheapest", "lowest cost", "least expensive", "lowest unit price", "least costly"]):
        col = "Cost Price Per Unit (USD)" if "Cost Price Per Unit (USD)" in df.columns else "Cost Price Total (USD)"
        if col in df.columns:
            display_cols = [c for c in ["Product ID", "Product Name", col] if c in df.columns]
            bottom = df.nsmallest(5, col)[display_cols]
            result["response"] = f"**Top 5 cheapest products by `{col}`:**"
            result["table_data"] = bottom
            result["tools_used"].append({
                "tool": "query_dataset", "args": {"operation": "bottom_n", "column": col, "n": 5},
                "result_summary": f"Bottom 5 by {col}", "success": True,
            })
        else:
            result["response"] = "Price column not found in dataset."

    # 7. Low stock / Reorder alerts / Out of stock
    elif any(w in msg for w in ["low stock", "lowest stock", "reorder", "out of stock", "least stock", "minimum stock", "shortage"]):
        if "Hand-In-Stock" in df.columns:
            display_cols = [c for c in ["Product ID", "Product Name", "Opening Stock", "Hand-In-Stock"] if c in df.columns]
            low_stock = df.nsmallest(10, "Hand-In-Stock")[display_cols]
            min_val = df["Hand-In-Stock"].min()
            result["response"] = (
                f"📦 **Top 10 Products with Lowest Stock on Hand:**\n"
                f"- Lowest current stock level: **{min_val} units**\n"
                f"- Review table below for items that may need replenishment."
            )
            result["table_data"] = low_stock
            result["tools_used"].append({
                "tool": "query_dataset", "args": {"operation": "bottom_n", "column": "Hand-In-Stock", "n": 10},
                "result_summary": f"10 items with lowest stock", "success": True,
            })
        else:
            result["response"] = "Hand-In-Stock column not found."

    # 8. Highest stock levels
    elif any(w in msg for w in ["highest stock", "most stock", "maximum stock", "largest stock", "excess stock"]):
        if "Hand-In-Stock" in df.columns:
            display_cols = [c for c in ["Product ID", "Product Name", "Hand-In-Stock"] if c in df.columns]
            high_stock = df.nlargest(10, "Hand-In-Stock")[display_cols]
            result["response"] = "**Top 10 Products with Highest Hand-In-Stock:**"
            result["table_data"] = high_stock
            result["tools_used"].append({
                "tool": "query_dataset", "args": {"operation": "top_n", "column": "Hand-In-Stock", "n": 10},
                "result_summary": "10 items with highest stock", "success": True,
            })
        else:
            result["response"] = "Hand-In-Stock column not found."

    # 9. Best selling / Units sold
    elif any(w in msg for w in ["best selling", "top selling", "most sold", "highest units sold", "highest sales", "popular"]):
        if "Number of Units Sold" in df.columns:
            display_cols = [c for c in ["Product ID", "Product Name", "Number of Units Sold", "Hand-In-Stock"] if c in df.columns]
            top_sold = df.nlargest(10, "Number of Units Sold")[display_cols]
            total_sold = df["Number of Units Sold"].sum()
            result["response"] = (
                f"🔥 **Top 10 Best-Selling Products by Units Sold:**\n"
                f"- **Total units sold across all products:** {total_sold:,} units"
            )
            result["table_data"] = top_sold
            result["tools_used"].append({
                "tool": "query_dataset", "args": {"operation": "top_n", "column": "Number of Units Sold", "n": 10},
                "result_summary": "Top 10 by units sold", "success": True,
            })
        else:
            result["response"] = "Units sold column not found."

    # 10. Least sold
    elif any(w in msg for w in ["least sold", "lowest sold", "worst selling", "slowest moving", "least units sold"]):
        if "Number of Units Sold" in df.columns:
            display_cols = [c for c in ["Product ID", "Product Name", "Number of Units Sold", "Hand-In-Stock"] if c in df.columns]
            least_sold = df.nsmallest(10, "Number of Units Sold")[display_cols]
            result["response"] = "**Top 10 Slowest-Moving Products (Lowest Units Sold):**"
            result["table_data"] = least_sold
            result["tools_used"].append({
                "tool": "query_dataset", "args": {"operation": "bottom_n", "column": "Number of Units Sold", "n": 10},
                "result_summary": "Bottom 10 by units sold", "success": True,
            })
        else:
            result["response"] = "Units sold column not found."

    # 11. Data Discrepancy / Formula Audit
    elif any(w in msg for w in ["discrepancy", "discrepancies", "mismatch", "inconsistency", "formula error", "audit", "math check"]):
        cols_needed = ["Opening Stock", "Purchase/Stock In", "Number of Units Sold", "Hand-In-Stock"]
        if all(c in df.columns for c in cols_needed):
            expected = df["Opening Stock"] + df["Purchase/Stock In"] - df["Number of Units Sold"]
            diff = df["Hand-In-Stock"] - expected
            mismatch_mask = diff != 0
            mismatch_df = df[mismatch_mask].copy()
            mismatch_df["Expected Stock"] = expected[mismatch_mask]
            mismatch_df["Difference"] = diff[mismatch_mask]

            display_cols = [c for c in ["Product ID", "Product Name", "Opening Stock", "Purchase/Stock In", "Number of Units Sold", "Hand-In-Stock", "Expected Stock", "Difference"] if c in mismatch_df.columns]
            result["response"] = (
                f"⚠️ **Inventory Formula Audit Found {len(mismatch_df)} Discrepancies:**\n\n"
                f"- Formula tested: `Hand-In-Stock = Opening Stock + Purchase - Units Sold`\n"
                f"- **{len(mismatch_df)} out of {len(df)} products** differ from the arithmetic expectation.\n"
                f"- This may indicate inventory shrinkage, damage write-offs, or unrecorded adjustments in the source data."
            )
            result["table_data"] = mismatch_df[display_cols]
            result["tools_used"].append({
                "tool": "query_dataset", "args": {"operation": "calculate_column"},
                "result_summary": f"{len(mismatch_df)} discrepancies found", "success": True,
            })
        else:
            result["response"] = "Required stock calculation columns not found in dataset."

    # 12. Correlation
    elif any(w in msg for w in ["correlation", "correlate", "relationship between"]):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 1:
            corr_df = df[num_cols].corr().round(3)
            result["response"] = "**Correlation Matrix Across Numeric Inventory Columns:**"
            result["table_data"] = corr_df
            result["tools_used"].append({
                "tool": "query_dataset", "args": {"operation": "correlation"},
                "result_summary": "Correlation matrix calculated", "success": True,
            })
        else:
            result["response"] = "Not enough numeric columns for correlation."

    # 13. Average / Mean statistics
    elif any(w in msg for w in ["average", "mean", "median"]) and not any(w in msg for w in ["what is", "define"]):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            means = df[num_cols].mean().round(2)
            lines = [f"- **{col}**: {val:,.2f}" for col, val in means.items()]
            result["response"] = "**Averages for Numeric Inventory Columns:**\n\n" + "\n".join(lines)
            result["tools_used"].append({
                "tool": "query_dataset", "args": {"operation": "aggregate", "function": "mean"},
                "result_summary": f"Means computed for {len(num_cols)} columns", "success": True,
            })
        else:
            result["response"] = "No numeric columns found."

    # 14. Product Search by Name or ID
    elif "Product Name" in df.columns or "Product ID" in df.columns:
        matched_rows = None
        # Check if any product name is explicitly mentioned in msg
        if "Product Name" in df.columns:
            for p_name in df["Product Name"].dropna().unique():
                if len(str(p_name)) > 2 and str(p_name).lower() in msg:
                    matched_rows = df[df["Product Name"].str.lower() == str(p_name).lower()]
                    break

        # Check for Product ID pattern like P101, P102, etc.
        if matched_rows is None or len(matched_rows) == 0:
            id_match = re.search(r'\b(p\d{2,4})\b', msg, re.IGNORECASE)
            if id_match and "Product ID" in df.columns:
                target_id = id_match.group(1).upper()
                matched_rows = df[df["Product ID"].str.upper() == target_id]

        if matched_rows is not None and len(matched_rows) > 0:
            result["response"] = f"Found product record for **{matched_rows.iloc[0].get('Product Name', 'Product')}**:"
            result["table_data"] = matched_rows
            result["tools_used"].append({
                "tool": "query_dataset", "args": {"operation": "search_rows"},
                "result_summary": f"Found {len(matched_rows)} matching record(s)", "success": True,
            })

    # 15. Preview / First few rows
    if not result["response"] and any(w in msg for w in ["show", "preview", "sample", "first", "head"]):
        n = 10
        result["response"] = f"Here are the first {n} rows of the inventory dataset:"
        result["table_data"] = df.head(n)
        result["tools_used"].append({
            "tool": "query_dataset", "args": {"operation": "head", "n": n},
            "result_summary": f"First {n} rows", "success": True,
        })

    # 16. Describe / Summary
    elif not result["response"] and any(w in msg for w in ["describe", "summary", "statistics", "stats", "overview"]):
        summary = get_dataset_summary(df)
        response_lines = [
            f"**Dataset Summary:**",
            f"- **Total products:** {summary['total_products']}",
            f"- **Total inventory value:** ${summary.get('total_inventory_value', 0):,.2f}",
            f"- **Average cost per unit:** ${summary.get('avg_cost_per_unit', 0):,.2f}",
            f"- **Total missing values:** {summary['total_missing']}",
        ]
        result["response"] = "\n".join(response_lines)
        result["table_data"] = df.describe(include="all")
        result["tools_used"].append({
            "tool": "query_dataset", "args": {"operation": "describe"},
            "result_summary": "Descriptive statistics", "success": True,
        })

    # 17. External Definitions / Web Search
    elif not result["response"] and any(w in msg for w in ["what is", "define", "definition of", "explain", "meaning of"]):
        search_res = web_search(user_message)
        if search_res.get("success") and search_res.get("results"):
            top = search_res["results"][0]
            result["response"] = (
                f"**{top['title']}**\n\n"
                f"{top['body']}\n\n"
                f"*Source: {top['href']}*"
            )
            result["tools_used"].append({
                "tool": "web_search", "args": {"query": user_message},
                "result_summary": f"Found {len(search_res['results'])} external results", "success": True,
            })

    # 18. Default Guidance
    if not result["response"]:
        result["response"] = (
            "You can ask me questions about your inventory data such as:\n\n"
            "- 📊 *How many products are in the dataset?*\n"
            "- 💰 *What is the total inventory value?*\n"
            "- 🏷️ *What are the most expensive or cheapest products?*\n"
            "- 📦 *Which products have the lowest stock or need reordering?*\n"
            "- 🔥 *What are the top selling products?*\n"
            "- ⚠️ *Are there any inventory formula discrepancies?*\n"
            "- 📋 *Show missing values or summary statistics*\n"
            "- 🔍 *Search for any specific product name or ID*"
        )

    return result
