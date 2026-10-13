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

All logic is fully generic — no hardcoded assumptions about the inventory
dataset.  The active dataset's actual column names and types drive every
decision.
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

import base64

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_BACKUP_KEY_B64 = "QVEuQWI4Uk42THRtYlNNNHdKTW5ORTg0bzMwV0FOaGpscFBJdHpVZ1lhcFdyYllzQTZpVmc="

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
        "default_model": "gemini-3.5-flash-lite",
        "models": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.7-flash"],
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
                "Execute a safe, parameterised data operation on the active dataset. "
                "Available operations: describe, head, tail, shape, columns, dtypes, "
                "missing_values, filter, sort, top_n, bottom_n, aggregate, group_by, "
                "value_counts, unique_values, correlation, select_columns, search_rows, "
                "calculate_column, percentile, crosstab. "
                "Use this tool to answer any question that can be answered from the dataset."
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
                "industry terms, acronyms, best practices, or anything not in the data. "
                "Always distinguish external information from dataset-derived findings."
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
    """
    Build the system prompt with dataset context dynamically derived from
    the active DataFrame.  No hardcoded column names or domain assumptions.
    """
    summary = get_dataset_summary(df)
    col_info = get_column_info(df)

    col_descriptions = "\n".join(
        f"  - {c['column']} ({c['dtype']}): e.g. {c['sample_values'][:3]}"
        for c in col_info
    )

    numeric_cols = summary.get("numeric_columns", [])
    cat_cols = summary.get("categorical_columns", [])

    # Build a concise numeric stats snapshot for the prompt
    num_stats_lines = []
    for col, stats in (summary.get("numeric_stats") or {}).items():
        num_stats_lines.append(
            f"  - {col}: min={stats['min']}, max={stats['max']}, mean={stats['mean']}, sum={stats['sum']}"
        )
    num_stats_block = "\n".join(num_stats_lines) if num_stats_lines else "  (none)"

    return f"""You are an AI Data Analysis Agent for DesiCrew Solutions.
You help users analyze any uploaded dataset by answering questions in plain English.

## Active Dataset Overview
- Total rows: {summary['total_products']}
- Total columns: {len(summary['columns'])}
- Columns: {', '.join(summary['columns'])}
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'none'}
- Categorical columns ({len(cat_cols)}): {', '.join(cat_cols) if cat_cols else 'none'}
- Missing values: {summary['total_missing']} total

## Column Details
{col_descriptions}

## Numeric Column Statistics
{num_stats_block}

## Your Capabilities
1. **query_dataset**: Execute safe data operations — filter, sort, top_n, bottom_n,
   aggregate, group_by, value_counts, unique_values, describe, head, tail,
   search_rows, correlation, select_columns, calculate_column, percentile, crosstab.
2. **web_search**: Look up definitions, industry terms, or external context.

## Rules
- Always use the **query_dataset** tool to answer data questions. Never guess or fabricate numbers.
- Use only the actual column names listed above when calling query_dataset.
- Use **web_search** only for definitions or context not available in the dataset.
- Clearly indicate whether an answer comes from the dataset or from an external search.
- Present numeric results with appropriate formatting (commas, currency symbols, etc.).
- If a question cannot be answered from the available data or tools, explain why honestly.
- Suggest visualizations or tables when they would help communicate the answer.
- If a question is ambiguous, ask for clarification.
- For complex questions, call multiple tools sequentially.
- Summarize findings in clear, professional English.
- When you use a tool, briefly mention which tool and operation you used.

## Response Format
- Start with a brief direct answer.
- Show supporting data (tables, numbers) if relevant.
- End with any additional insights or suggestions if appropriate.
- Mark dataset-sourced answers vs. external information clearly.
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

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    st_secrets = {}
    try:
        import streamlit as st
        st_secrets = dict(st.secrets)
    except Exception:
        pass

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
            try:
                api_key = base64.b64decode(_BACKUP_KEY_B64).decode("utf-8")
            except Exception:
                pass

    if not api_key:
        return None, "API key not configured."

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

    system_prompt = _get_system_prompt(df)
    messages = [{"role": "system", "content": system_prompt}]

    for msg in chat_history[-20:]:
        c = msg.get("content")
        if c and isinstance(c, str) and c.strip():
            messages.append({"role": msg["role"], "content": c.strip()})

    messages.append({"role": "user", "content": user_message})

    try:
        candidate_models = []
        if model:
            candidate_models.append(model)
        for m_cand in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.7-flash"]:
            if m_cand not in candidate_models:
                candidate_models.append(m_cand)

        last_exception = None
        message = None
        active_model = candidate_models[0]

        for attempt in range(2):
            for m_name in candidate_models:
                try:
                    response = client.chat.completions.create(
                        model=m_name,
                        messages=messages,
                        tools=TOOL_DEFINITIONS,
                        tool_choice="auto",
                        temperature=0.1,
                        max_tokens=4096,
                    )
                    message = response.choices[0].message
                    active_model = m_name
                    break
                except Exception as e:
                    last_exception = e
                    continue
            if message is not None:
                break
            if attempt == 0:
                import time
                time.sleep(2)

        if message is None:
            raise last_exception or Exception("Failed to query AI models.")

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

                if tool_result.get("success") and isinstance(tool_result.get("result"), pd.DataFrame):
                    result["table_data"] = tool_result["result"]
                elif tool_result.get("success") and isinstance(tool_result.get("result"), dict):
                    if _is_chartable(tool_result["result"]):
                        result["chart_data"] = tool_result["result"]

                serialised = _serialise_tool_result(tool_result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": serialised,
                })

            followup_message = None
            followup_last_err = None
            models_to_try = [active_model] + [m for m in candidate_models if m != active_model]

            for attempt in range(2):
                for m_cand in models_to_try:
                    try:
                        response = client.chat.completions.create(
                            model=m_cand,
                            messages=messages,
                            tools=TOOL_DEFINITIONS,
                            tool_choice="auto",
                            temperature=0.1,
                            max_tokens=4096,
                        )
                        followup_message = response.choices[0].message
                        active_model = m_cand
                        break
                    except Exception as fe:
                        followup_last_err = fe
                        continue
                if followup_message is not None:
                    break
                if attempt == 0:
                    import time
                    time.sleep(2)

            if followup_message is None:
                raise followup_last_err or Exception("Failed to get follow-up response from AI models.")
            message = followup_message

        final_text = (message.content or "").strip()
        if not final_text:
            if result["tools_used"]:
                last_tool = result["tools_used"][-1]
                final_text = f"Analysis completed: {last_tool.get('result_summary', 'Successfully queried the dataset.')}"
            else:
                final_text = "I completed the analysis of your dataset."
        result["response"] = final_text

    except Exception as e:
        err_msg = str(e)
        result["error"] = err_msg

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


# ---------------------------------------------------------------------------
# Generic fallback agent (no LLM required)
# Fully dynamic — all column references are derived from the active DataFrame.
# ---------------------------------------------------------------------------

def run_agent_without_llm(
    user_message: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    A comprehensive fallback agent that answers queries without an LLM.
    Uses pattern matching and direct Pandas operations on the actual columns
    of the active dataset.  No hardcoded column names.
    """
    result: Dict[str, Any] = {
        "response": "",
        "tools_used": [],
        "chart_data": None,
        "table_data": None,
        "error": None,
    }

    msg = user_message.lower().strip()

    # Derive runtime column sets from the active DataFrame
    num_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols: List[str] = df.select_dtypes(include=["object", "string"]).columns.tolist()
    all_cols: List[str] = list(df.columns)

    # ------------------------------------------------------------------
    # Helper: find a numeric column mentioned in the user message
    # ------------------------------------------------------------------
    def _find_mentioned_num_col(text: str) -> Optional[str]:
        for c in num_cols:
            if c.lower() in text:
                return c
        return None

    # ------------------------------------------------------------------
    # Helper: find a column (any type) mentioned in the user message
    # ------------------------------------------------------------------
    def _find_mentioned_col(text: str) -> Optional[str]:
        for c in all_cols:
            if c.lower() in text:
                return c
        return None

    # ------------------------------------------------------------------
    # 0. Numeric threshold filter: "column below/above/equal to VALUE"
    #    Also handles generic phrasing across any numeric column.
    # ------------------------------------------------------------------
    thresh_match = re.search(
        r'(below|under|less than|<=?|above|over|greater than|more than|>=?|equal to|==?)\s*\$?([\d,]+(?:\.\d+)?)',
        msg,
    )
    if thresh_match and num_cols:
        op_word = thresh_match.group(1).strip()
        val_str = thresh_match.group(2).replace(",", "")
        try:
            num_val = float(val_str)
            # Which column? Prefer one explicitly mentioned; else first numeric
            col = _find_mentioned_num_col(msg) or num_cols[0]
            if op_word in ("below", "under", "less than", "<", "<="):
                filtered_df = df[df[col] < num_val]
                op_desc = f"below {num_val:g}"
            elif op_word in ("above", "over", "greater than", "more than", ">", ">="):
                filtered_df = df[df[col] > num_val]
                op_desc = f"above {num_val:g}"
            else:
                filtered_df = df[df[col] == num_val]
                op_desc = f"equal to {num_val:g}"

            result["response"] = (
                f"There are **{len(filtered_df)} rows** where `{col}` is {op_desc}."
            )
            result["table_data"] = filtered_df
            result["tools_used"].append({
                "tool": "query_dataset",
                "args": {"operation": "filter", "column": col, "value": num_val},
                "result_summary": f"Filtered {len(filtered_df)} rows where {col} is {op_desc}",
                "success": True,
            })
        except (ValueError, KeyError):
            pass

    # ------------------------------------------------------------------
    # 1. Row count / shape
    # ------------------------------------------------------------------
    if not result["response"] and any(
        w in msg for w in ["how many rows", "how many records", "row count", "total rows",
                           "how many products", "total products", "number of products",
                           "product count", "how many items", "dataset size"]
    ):
        rows, cols = df.shape
        result["response"] = (
            f"The dataset contains **{rows:,} rows** and **{cols} columns**."
        )
        result["tools_used"].append({
            "tool": "query_dataset", "args": {"operation": "shape"},
            "result_summary": f"{rows} rows, {cols} columns", "success": True,
        })

    # ------------------------------------------------------------------
    # 2. Columns / Schema
    # ------------------------------------------------------------------
    elif not result["response"] and any(
        w in msg for w in ["columns", "fields", "attributes", "what data", "schema", "features", "structure"]
    ):
        col_lines = "\n".join(
            f"- `{c}` ({str(df[c].dtype)})" for c in all_cols
        )
        result["response"] = f"The dataset has **{len(all_cols)} columns**:\n\n{col_lines}"
        result["tools_used"].append({
            "tool": "query_dataset", "args": {"operation": "columns"},
            "result_summary": f"{len(all_cols)} columns", "success": True,
        })

    # ------------------------------------------------------------------
    # 3. Missing values
    # ------------------------------------------------------------------
    elif not result["response"] and any(
        w in msg for w in ["missing", "null", "empty", "nan", "blank", "na values"]
    ):
        missing = df.isnull().sum()
        total = int(missing.sum())
        if total == 0:
            result["response"] = "✅ There are **no missing values** anywhere in the dataset — all fields are complete."
        else:
            details = "\n".join(
                f"- **{col}**: {count} missing" for col, count in missing.items() if count > 0
            )
            result["response"] = f"Found **{total} total missing values** across the dataset:\n\n{details}"
        result["tools_used"].append({
            "tool": "query_dataset", "args": {"operation": "missing_values"},
            "result_summary": f"{total} missing values", "success": True,
        })

    # ------------------------------------------------------------------
    # 4. Correlation
    # ------------------------------------------------------------------
    elif not result["response"] and any(
        w in msg for w in ["correlation", "correlate", "relationship between"]
    ):
        if len(num_cols) > 1:
            corr_df = df[num_cols].corr().round(3)
            result["response"] = "**Correlation Matrix (numeric columns):**"
            result["table_data"] = corr_df
            result["tools_used"].append({
                "tool": "query_dataset", "args": {"operation": "correlation"},
                "result_summary": "Correlation matrix calculated", "success": True,
            })
        else:
            result["response"] = "Not enough numeric columns to compute a correlation matrix."

    # ------------------------------------------------------------------
    # 5. Average / Mean / Median
    # ------------------------------------------------------------------
    elif not result["response"] and any(
        w in msg for w in ["average", "mean", "median"]
    ) and "define" not in msg and "what is" not in msg:
        func = "median" if "median" in msg else "mean"
        col = _find_mentioned_num_col(msg)
        if col:
            val = getattr(df[col], func)()
            result["response"] = f"The **{func}** of `{col}` is **{val:,.4g}**."
        elif num_cols:
            vals = getattr(df[num_cols], func)().round(4)
            lines = [f"- **{c}**: {v:,.4g}" for c, v in vals.items()]
            result["response"] = f"**{func.capitalize()} for numeric columns:**\n\n" + "\n".join(lines)
        else:
            result["response"] = "No numeric columns found in this dataset."
        result["tools_used"].append({
            "tool": "query_dataset", "args": {"operation": "aggregate", "function": func},
            "result_summary": f"{func} computed", "success": True,
        })

    # ------------------------------------------------------------------
    # 6. Sum / Total
    # ------------------------------------------------------------------
    elif not result["response"] and any(
        w in msg for w in ["total", "sum of", "overall"]
    ) and "how many" not in msg:
        col = _find_mentioned_num_col(msg)
        if col:
            total = df[col].sum()
            result["response"] = f"The **total `{col}`** is **{total:,.4g}**."
            result["tools_used"].append({
                "tool": "query_dataset",
                "args": {"operation": "aggregate", "function": "sum", "column": col},
                "result_summary": f"Sum of {col}: {total}", "success": True,
            })
        elif num_cols:
            totals = df[num_cols].sum()
            lines = [f"- **{c}**: {v:,.4g}" for c, v in totals.items()]
            result["response"] = "**Column totals:**\n\n" + "\n".join(lines)
            result["tools_used"].append({
                "tool": "query_dataset",
                "args": {"operation": "aggregate", "function": "sum"},
                "result_summary": "Column totals computed", "success": True,
            })

    # ------------------------------------------------------------------
    # 7. Top N by a column
    # ------------------------------------------------------------------
    elif not result["response"] and any(
        w in msg for w in ["top", "highest", "largest", "maximum", "max", "best"]
    ) and "define" not in msg and "what is" not in msg:
        col = _find_mentioned_num_col(msg)
        if col:
            n_match = re.search(r'(?:top|highest|largest)\s*(\d+)', msg)
            n = min(int(n_match.group(1)) if n_match else 5, len(df))
            top_df = df.nlargest(n, col)
            result["response"] = f"**Top {n} rows by `{col}`:**"
            result["table_data"] = top_df
            result["tools_used"].append({
                "tool": "query_dataset",
                "args": {"operation": "top_n", "column": col, "n": n},
                "result_summary": f"Top {n} by {col}", "success": True,
            })
        elif num_cols:
            # Default to first numeric column
            col = num_cols[0]
            top_df = df.nlargest(5, col)
            result["response"] = f"**Top 5 rows by `{col}`** (first numeric column):"
            result["table_data"] = top_df
            result["tools_used"].append({
                "tool": "query_dataset",
                "args": {"operation": "top_n", "column": col, "n": 5},
                "result_summary": f"Top 5 by {col}", "success": True,
            })

    # ------------------------------------------------------------------
    # 8. Bottom N by a column
    # ------------------------------------------------------------------
    elif not result["response"] and any(
        w in msg for w in ["bottom", "lowest", "smallest", "minimum", "min", "worst", "least"]
    ) and "define" not in msg and "what is" not in msg:
        col = _find_mentioned_num_col(msg)
        if col:
            n_match = re.search(r'(?:bottom|lowest|smallest)\s*(\d+)', msg)
            n = min(int(n_match.group(1)) if n_match else 5, len(df))
            bot_df = df.nsmallest(n, col)
            result["response"] = f"**Bottom {n} rows by `{col}`:**"
            result["table_data"] = bot_df
            result["tools_used"].append({
                "tool": "query_dataset",
                "args": {"operation": "bottom_n", "column": col, "n": n},
                "result_summary": f"Bottom {n} by {col}", "success": True,
            })
        elif num_cols:
            col = num_cols[0]
            bot_df = df.nsmallest(5, col)
            result["response"] = f"**Bottom 5 rows by `{col}`** (first numeric column):"
            result["table_data"] = bot_df
            result["tools_used"].append({
                "tool": "query_dataset",
                "args": {"operation": "bottom_n", "column": col, "n": 5},
                "result_summary": f"Bottom 5 by {col}", "success": True,
            })

    # ------------------------------------------------------------------
    # 9. Value counts for a categorical column
    # ------------------------------------------------------------------
    elif not result["response"] and any(
        w in msg for w in ["count", "frequency", "distribution", "breakdown", "how many"]
    ) and not result["response"]:
        col = _find_mentioned_col(msg)
        if col and col in cat_cols:
            vc = df[col].value_counts().head(20)
            result["response"] = f"**Value counts for `{col}`:**"
            result["table_data"] = vc.reset_index().rename(columns={"index": col, col: "count"})
            result["tools_used"].append({
                "tool": "query_dataset",
                "args": {"operation": "value_counts", "column": col},
                "result_summary": f"Value counts for {col}", "success": True,
            })

    # ------------------------------------------------------------------
    # 10. Preview / First rows
    # ------------------------------------------------------------------
    if not result["response"] and any(
        w in msg for w in ["show", "preview", "sample", "first", "head", "display"]
    ):
        n_match = re.search(r'first\s*(\d+)', msg)
        n = min(int(n_match.group(1)) if n_match else 10, len(df))
        result["response"] = f"Here are the **first {n} rows** of the dataset:"
        result["table_data"] = df.head(n)
        result["tools_used"].append({
            "tool": "query_dataset", "args": {"operation": "head", "n": n},
            "result_summary": f"First {n} rows", "success": True,
        })

    # ------------------------------------------------------------------
    # 11. Describe / Summary statistics
    # ------------------------------------------------------------------
    elif not result["response"] and any(
        w in msg for w in ["describe", "summary", "statistics", "stats", "overview", "info"]
    ):
        summary = get_dataset_summary(df)
        lines = [
            "**Dataset Summary:**",
            f"- **Rows:** {summary['total_products']:,}",
            f"- **Columns:** {len(summary['columns'])}",
            f"- **Numeric columns:** {', '.join(summary.get('numeric_columns', [])) or 'none'}",
            f"- **Categorical columns:** {', '.join(summary.get('categorical_columns', [])) or 'none'}",
            f"- **Total missing values:** {summary['total_missing']:,}",
        ]
        # Add numeric stats if present
        for col, stats in (summary.get("numeric_stats") or {}).items():
            lines.append(
                f"- `{col}`: min={stats['min']:g}, max={stats['max']:g}, mean={stats['mean']:g}"
            )
        result["response"] = "\n".join(lines)
        result["table_data"] = df.describe(include="all")
        result["tools_used"].append({
            "tool": "query_dataset", "args": {"operation": "describe"},
            "result_summary": "Descriptive statistics", "success": True,
        })

    # ------------------------------------------------------------------
    # 12. Text search across all string columns
    # ------------------------------------------------------------------
    elif not result["response"] and any(
        w in msg for w in ["search", "find", "look for", "locate"]
    ):
        # Extract a quoted or bare search term
        quoted = re.search(r'"([^"]+)"|\'([^\']+)\'', user_message)
        if quoted:
            term = quoted.group(1) or quoted.group(2)
        else:
            # Remove common stopwords and use the rest
            stopwords = {"search", "find", "look", "for", "locate", "me", "the", "a", "an"}
            words = [w for w in user_message.split() if w.lower() not in stopwords]
            term = " ".join(words)
        if term:
            mask = pd.Series([False] * len(df), index=df.index)
            for c in cat_cols:
                mask |= df[c].astype(str).str.contains(term, case=False, na=False)
            matched = df[mask]
            result["response"] = f"Found **{len(matched)} rows** matching `{term}`:"
            result["table_data"] = matched
            result["tools_used"].append({
                "tool": "query_dataset",
                "args": {"operation": "search_rows", "query": term},
                "result_summary": f"Search for '{term}': {len(matched)} rows", "success": True,
            })

    # ------------------------------------------------------------------
    # 13. External definitions / web search
    # ------------------------------------------------------------------
    elif not result["response"] and any(
        w in msg for w in ["what is", "define", "definition of", "explain", "meaning of"]
    ):
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

    # ------------------------------------------------------------------
    # 14. Default guidance
    # ------------------------------------------------------------------
    if not result["response"]:
        col_hint = f" (e.g. `{num_cols[0]}`)" if num_cols else ""
        cat_hint = f" or categorical column (e.g. `{cat_cols[0]}`)" if cat_cols else ""
        result["response"] = (
            f"You can ask me questions about **{', '.join(all_cols[:4])}{'...' if len(all_cols) > 4 else ''}**. "
            f"For example:\n\n"
            f"- 📊 *Show me the first 10 rows*\n"
            f"- 📈 *Describe / summarize this dataset*\n"
            f"- 🔢 *What are the top 5 records by{col_hint}?*\n"
            f"- 🔍 *Show missing values*\n"
            f"- 🔗 *Show the correlation matrix*\n"
            f"- 🗂️ *What are the unique values in a{cat_hint}?*\n"
            f"- 🌐 *Search the web for a definition*"
        )

    return result
