"""
app.py — DesiCrew AI Data Agent
Streamlit web application for interactive analysis of arbitrary Excel / CSV datasets.

DesiCrew Solutions — Data Science Internship Evaluation — Question 1
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from pathlib import Path

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.data_loader import (
    load_dataset,
    load_inventory_excel,
    get_dataset_summary,
    get_column_info,
    inspect_excel_sheets,
)
from src.agent import run_agent, run_agent_without_llm, get_api_client, PROVIDER_CONFIGS
from src.safe_executor import get_available_operations


# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="DesiCrew AI Data Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ─── Global ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #cbd5e1 !important;
    }

    /* ─── Metric cards ─── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    div[data-testid="stMetric"] label {
        color: #475569 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* ─── Chat bubbles ─── */
    .chat-user {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        padding: 14px 20px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 85%;
        margin-left: auto;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
    }
    .chat-assistant {
        background: #f8fafc;
        color: #1e293b;
        padding: 14px 20px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        border: 1px solid #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .chat-assistant strong { color: #0f172a; }

    /* ─── Tool badge ─── */
    .tool-badge {
        display: inline-block;
        background: #f0fdf4;
        color: #15803d;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
        border: 1px solid #bbf7d0;
    }
    .tool-badge-error {
        background: #fef2f2;
        color: #dc2626;
        border-color: #fecaca;
    }

    /* ─── Tool details expander ─── */
    .tool-details {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0 12px 0;
        font-size: 0.82rem;
        color: #475569;
    }

    /* ─── Header bar ─── */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 0.9rem;
        color: #94a3b8;
    }

    /* ─── Status indicator ─── */
    .status-ok {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f0fdf4;
        color: #15803d;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #bbf7d0;
    }
    .status-warn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #fffbeb;
        color: #b45309;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #fde68a;
    }
    .status-error {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #fef2f2;
        color: #dc2626;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #fecaca;
    }

    /* ─── Upload area ─── */
    div[data-testid="stFileUploader"] > section {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        background: #f8fafc !important;
    }
    div[data-testid="stFileUploader"] > section:hover {
        border-color: #3b82f6 !important;
        background: #eff6ff !important;
    }

    /* ─── Data table tweaks ─── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* ─── Suggested question buttons ─── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }

    /* ─── Tab styling ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }

    /* ─── Sheet selector banner ─── */
    .sheet-banner {
        background: linear-gradient(90deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        color: #0369a1;
    }

    /* ─── Hide Streamlit branding ─── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────
def _init_session_state():
    defaults = {
        "chat_history": [],
        "df": None,
        "meta": None,
        "api_status": None,
        "dataset_name": None,
        "last_uploaded_file_id": None,
        # Multi-sheet support
        "available_sheets": [],
        "active_sheet": None,
        "pending_file_buffer": None,   # raw bytes of last uploaded Excel
        "pending_file_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session_state()


# ── Helper: load file and reset context ────────────────────────────────────

def _activate_dataset(
    file_path_or_buffer,
    filename: str,
    file_id: str,
    sheet_name: str = None,
    silent: bool = False,
):
    """
    Load file_path_or_buffer as the active dataset and fully reset chat context.
    On success, updates session_state and appends a welcome message.
    Returns True on success, False on failure (error shown via st.error).
    """
    try:
        new_df, new_meta = load_dataset(
            file_path_or_buffer, filename=filename, sheet_name=sheet_name
        )
        # --- Context isolation: clear previous chat history ---
        st.session_state.chat_history = []
        st.session_state.df = new_df
        st.session_state.meta = new_meta
        st.session_state.dataset_name = filename
        st.session_state.last_uploaded_file_id = file_id
        st.session_state.available_sheets = new_meta.get("sheets", [])
        st.session_state.active_sheet = new_meta.get("active_sheet")

        cols_preview = ", ".join(f"`{c}`" for c in new_df.columns[:8])
        if len(new_df.columns) > 8:
            cols_preview += f" ... (+{len(new_df.columns) - 8} more)"

        sheet_line = ""
        if st.session_state.available_sheets and len(st.session_state.available_sheets) > 1:
            sheet_line = (
                f"\n- **Sheets available:** "
                f"{', '.join(f'`{s}`' for s in st.session_state.available_sheets)}"
                f"\n- **Active sheet:** `{st.session_state.active_sheet}`"
            )

        if not silent:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": (
                    f"📁 **Dataset Activated:** `{filename}`\n\n"
                    f"- **Rows:** {len(new_df):,}\n"
                    f"- **Columns ({len(new_df.columns)}):** {cols_preview}"
                    f"{sheet_line}\n\n"
                    f"Chat history has been reset. Ask me anything about `{filename}`!"
                ),
            })
        return True
    except Exception as e:
        if not silent:
            st.error(f"❌ Error loading file: {e}")
        return False


def auto_load_default_dataset():
    """Attempt to load the bundled DesiCrew inventory dataset automatically on first run."""
    default_path = Path(__file__).parent / "Inventory-Records-Sample-Data.xlsx"
    if default_path.exists() and st.session_state.df is None:
        file_id = f"default_{default_path.stat().st_mtime}"
        _activate_dataset(str(default_path), "Inventory-Records-Sample-Data.xlsx", file_id, silent=True)
        return True
    return st.session_state.df is not None


def check_api_status():
    """Check if the AI API is configured and reachable."""
    client, result = get_api_client()
    if client is None:
        st.session_state.api_status = {"ok": False, "message": result}
        return False
    st.session_state.api_status = {"ok": True, "client": client, "model": result}
    return True


def render_chat_message(role: str, content: str, tools_used=None):
    """Render a chat message using native Streamlit chat bubbles."""
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if tools_used:
            for tool in tools_used:
                css_class = "tool-badge" if tool.get("success") else "tool-badge tool-badge-error"
                icon = "✅" if tool.get("success") else "❌"
                tool_name = tool.get("tool", "unknown")
                st.markdown(
                    f'<span class="{css_class}">{icon} {tool_name}</span>',
                    unsafe_allow_html=True,
                )
            with st.expander("🔍 Tool execution details", expanded=False):
                for tool in tools_used:
                    st.markdown(f"""<div class="tool-details">
                        <strong>Tool:</strong> {tool.get('tool', 'N/A')}<br>
                        <strong>Args:</strong> <code>{tool.get('args', {})}</code><br>
                        <strong>Result:</strong> {tool.get('result_summary', 'N/A')}<br>
                        <strong>Status:</strong> {'✅ Success' if tool.get('success') else '❌ Failed'}
                    </div>""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 DesiCrew AI Data Agent")
    st.markdown("---")

    # AI Agent status
    check_api_status()
    api = st.session_state.api_status
    if api and api["ok"]:
        st.markdown(
            '<div class="status-ok">🟢 AI Agent Active</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-warn">🟡 Offline Mode (no API key)</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Auto-load default dataset on first run
    auto_load_default_dataset()

    # Dataset status
    if st.session_state.df is not None:
        df_sidebar = st.session_state.df
        ds_name = st.session_state.get("dataset_name", "Dataset")
        st.markdown(
            f'<div class="status-ok">📁 {ds_name} — {len(df_sidebar):,} rows × {len(df_sidebar.columns)} cols</div>',
            unsafe_allow_html=True,
        )
        # Show active sheet if multi-sheet workbook
        sheets = st.session_state.get("available_sheets", [])
        if sheets and len(sheets) > 1:
            st.markdown(f"**Sheet:** `{st.session_state.get('active_sheet', sheets[0])}`")
    else:
        st.markdown(
            '<div class="status-error">📁 No Dataset Loaded</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── File upload ────────────────────────────────────────────────────────
    st.markdown("### 📤 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload any Excel or CSV dataset",
        type=["xlsx", "xls", "csv", "tsv"],
        key="sidebar_file_uploader",
        help="Supports .xlsx, .xls (with multi-sheet selection), .csv, .tsv",
    )

    if uploaded_file is not None:
        file_id = f"side_{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_uploaded_file_id") != file_id:
            # Peek at sheets before loading
            raw_bytes = uploaded_file.read()
            import io
            sheets = inspect_excel_sheets(io.BytesIO(raw_bytes))
            if len(sheets) > 1:
                # Store buffer and show sheet selector (handled in main area)
                st.session_state.pending_file_buffer = raw_bytes
                st.session_state.pending_file_name = uploaded_file.name
                st.session_state.pending_file_id = file_id
                st.session_state.pending_file_sheets = sheets
                st.info(f"📋 **{len(sheets)} sheets found** — select a sheet below to load.")
            else:
                # Single-sheet or CSV: load immediately
                ok = _activate_dataset(
                    io.BytesIO(raw_bytes),
                    uploaded_file.name,
                    file_id,
                )
                if ok:
                    st.success(f"✅ Loaded {len(st.session_state.df):,} rows × {len(st.session_state.df.columns)} columns")
                    st.rerun()

    # Sheet selector for multi-sheet uploads (sidebar)
    if st.session_state.get("pending_file_buffer") and st.session_state.get("last_uploaded_file_id") != st.session_state.get("pending_file_id"):
        sheets = st.session_state.get("pending_file_sheets", [])
        chosen_sheet = st.selectbox(
            "Select worksheet to load:",
            options=sheets,
            key="sidebar_sheet_selector",
        )
        if st.button("✅ Load Selected Sheet", key="sidebar_load_sheet_btn", use_container_width=True):
            import io as _io
            ok = _activate_dataset(
                _io.BytesIO(st.session_state.pending_file_buffer),
                st.session_state.pending_file_name,
                st.session_state.pending_file_id,
                sheet_name=chosen_sheet,
            )
            if ok:
                st.session_state.pending_file_buffer = None
                st.success(f"✅ Loaded sheet '{chosen_sheet}'")
                st.rerun()

    st.markdown("---")

    # Reset to default
    if st.button("🔄 Reset to Default Dataset", use_container_width=True, key="sidebar_reset_btn"):
        st.session_state.df = None
        st.session_state.meta = None
        st.session_state.dataset_name = None
        st.session_state.last_uploaded_file_id = None
        st.session_state.available_sheets = []
        st.session_state.active_sheet = None
        st.session_state.pending_file_buffer = None
        auto_load_default_dataset()
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "🔄 Reset to default DesiCrew inventory dataset. Chat history cleared.",
        }]
        st.rerun()

    # Clear chat history
    if st.button("🗑️ Clear Chat History", use_container_width=True, key="sidebar_clear_chat_btn"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem; color:#64748b; text-align:center;'>"
        "DesiCrew Solutions<br>Data Science Internship — Q1<br>"
        "AI Data Agent — Universal Excel / CSV</p>",
        unsafe_allow_html=True,
    )


# ── Main content area ─────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 AI Data Agent</h1>
    <p>Ask questions about any uploaded Excel or CSV dataset in plain English.
    Powered by AI with safe, validated data operations.</p>
</div>
""", unsafe_allow_html=True)


# If no dataset loaded, show upload prompt
if st.session_state.df is None:
    st.warning("⬆️ **Please upload an Excel or CSV dataset** using the sidebar to get started.")
    st.info(
        "This agent works with **any Excel (.xlsx / .xls) or CSV file**. "
        "Multi-sheet workbooks are supported — you can choose which sheet to analyse. "
        "The bundled DesiCrew inventory file is loaded automatically if present."
    )
    st.stop()

df = st.session_state.df

# ── Sheet switcher (shown when workbook has multiple sheets) ───────────────
sheets = st.session_state.get("available_sheets", [])
if sheets and len(sheets) > 1:
    active = st.session_state.get("active_sheet", sheets[0])
    st.markdown(
        f'<div class="sheet-banner">📑 <strong>Multi-sheet workbook</strong> — '
        f'currently viewing sheet <strong>{active}</strong>. '
        f'Switch below to analyse a different sheet.</div>',
        unsafe_allow_html=True,
    )
    col_sw1, col_sw2 = st.columns([3, 1])
    with col_sw1:
        new_sheet = st.selectbox(
            "Switch worksheet:",
            options=sheets,
            index=sheets.index(active) if active in sheets else 0,
            key="main_sheet_selector",
            label_visibility="collapsed",
        )
    with col_sw2:
        if st.button("🔄 Switch Sheet", key="main_switch_sheet_btn"):
            buf = st.session_state.get("pending_file_buffer")
            fname = st.session_state.get("pending_file_name") or st.session_state.get("dataset_name", "")
            fid = st.session_state.get("pending_file_id") or st.session_state.get("last_uploaded_file_id", "")
            if buf:
                import io as _io
                ok = _activate_dataset(
                    _io.BytesIO(buf),
                    fname,
                    fid + f"_sheet_{new_sheet}",
                    sheet_name=new_sheet,
                )
                if ok:
                    st.rerun()
            else:
                # Fallback for default dataset (single sheet)
                st.warning("Sheet switching requires the file buffer. Please re-upload the file.")


# ── Summary metrics row ───────────────────────────────────────────────────
summary = get_dataset_summary(df)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Total Rows", f"{len(df):,}")

with m2:
    # Show inventory value if column exists, else first numeric sum
    if "total_inventory_value" in summary:
        val = summary["total_inventory_value"]
        st.metric("Inventory Value", f"${val:,.0f}")
    elif numeric_cols:
        col0 = numeric_cols[0]
        col0_sum = df[col0].sum()
        lbl = col0[:12] + ("…" if len(col0) > 12 else "")
        st.metric(f"Total {lbl}", f"{col0_sum:,.2f}")
    else:
        st.metric("Columns", f"{len(df.columns)}")

with m3:
    # Show avg unit cost if column exists, else mean of first numeric column
    if "avg_cost_per_unit" in summary:
        avg = summary["avg_cost_per_unit"]
        st.metric("Avg Unit Cost", f"${avg:,.2f}")
    elif numeric_cols:
        col0 = numeric_cols[0]
        col0_mean = df[col0].mean()
        lbl = col0[:12] + ("…" if len(col0) > 12 else "")
        st.metric(f"Avg {lbl}", f"{col0_mean:,.2f}" if pd.notna(col0_mean) else "N/A")
    else:
        st.metric("Total Cells", f"{df.size:,}")

with m4:
    # Second numeric column sum, or column count
    if len(numeric_cols) >= 2:
        col1 = numeric_cols[1]
        col1_sum = df[col1].sum()
        lbl = col1[:12] + ("…" if len(col1) > 12 else "")
        st.metric(f"Total {lbl}", f"{col1_sum:,.2f}")
    else:
        st.metric("Columns", f"{len(df.columns)}")

with m5:
    st.metric("Missing Values", f"{summary['total_missing']:,}")

st.markdown("")

# ── Tabs: Chat | Data Preview | Analytics ─────────────────────────────────
tab_chat, tab_preview, tab_analytics = st.tabs(["💬 Chat", "📋 Data Preview", "📈 Analytics"])


# ── Tab: Chat ──────────────────────────────────────────────────────────────
with tab_chat:

    # ── In-Chat Dataset Uploader ───────────────────────────────────────────
    with st.expander(
        "📎 Upload or Switch Dataset (.csv, .xlsx, .xls)",
        expanded=(st.session_state.df is None),
    ):
        up_col1, up_col2 = st.columns([3, 1])
        with up_col1:
            chat_uploaded_file = st.file_uploader(
                "Upload any dataset to analyse in this chat",
                type=["xlsx", "xls", "csv", "tsv"],
                key="chat_file_uploader",
                help="Upload any .csv, .xlsx, or .xls file. Chat history is reset to avoid contamination.",
            )
        with up_col2:
            cur_name = st.session_state.get("dataset_name", "Default Sample Dataset")
            active_sheet_label = st.session_state.get("active_sheet", "")
            if active_sheet_label:
                cur_name += f"\n(sheet: {active_sheet_label})"
            st.markdown(f"**Active Dataset:**<br>`{cur_name}`", unsafe_allow_html=True)
            if st.button("🔄 Reset to Default", key="reset_to_default_btn", use_container_width=True):
                st.session_state.df = None
                st.session_state.meta = None
                st.session_state.dataset_name = None
                st.session_state.last_uploaded_file_id = None
                st.session_state.available_sheets = []
                st.session_state.active_sheet = None
                st.session_state.pending_file_buffer = None
                auto_load_default_dataset()
                st.session_state.chat_history = [{
                    "role": "assistant",
                    "content": "🔄 Reset to default DesiCrew inventory dataset. Chat history cleared.",
                }]
                st.rerun()

    if chat_uploaded_file is not None:
        file_id = f"chat_{chat_uploaded_file.name}_{chat_uploaded_file.size}"
        if st.session_state.get("last_uploaded_file_id") != file_id:
            import io
            raw_bytes = chat_uploaded_file.read()
            sheets_found = inspect_excel_sheets(io.BytesIO(raw_bytes))

            if len(sheets_found) > 1:
                # Show sheet selector inside the expander
                st.markdown(f"📑 **{len(sheets_found)} sheets found in `{chat_uploaded_file.name}`:**")
                chosen = st.selectbox(
                    "Select worksheet to load:",
                    options=sheets_found,
                    key="chat_sheet_selector",
                )
                if st.button("✅ Load Selected Sheet", key="chat_load_sheet_btn"):
                    ok = _activate_dataset(
                        io.BytesIO(raw_bytes),
                        chat_uploaded_file.name,
                        file_id,
                        sheet_name=chosen,
                    )
                    if ok:
                        st.session_state.pending_file_buffer = raw_bytes
                        st.session_state.pending_file_name = chat_uploaded_file.name
                        st.session_state.pending_file_id = file_id
                        st.session_state.pending_file_sheets = sheets_found
                        st.success(f"✅ Loaded sheet '{chosen}' from `{chat_uploaded_file.name}`")
                        st.rerun()
            else:
                ok = _activate_dataset(
                    io.BytesIO(raw_bytes),
                    chat_uploaded_file.name,
                    file_id,
                )
                if ok:
                    st.session_state.pending_file_buffer = raw_bytes
                    st.session_state.pending_file_name = chat_uploaded_file.name
                    st.session_state.pending_file_id = file_id
                    st.session_state.pending_file_sheets = sheets_found
                    st.rerun()

    # ── Dynamic Suggested Questions ────────────────────────────────────────
    st.markdown("**💡 Try asking:**")
    q_cols = st.columns(4)

    # Build suggestions dynamically from the active DataFrame
    suggestions = []
    if numeric_cols:
        c0 = numeric_cols[0]
        c1 = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        suggestions.append(f"What are the top 5 records by {c0}?")
        suggestions.append(f"What is the average {c0}?")
    if cat_cols:
        suggestions.append(f"Show value counts for {cat_cols[0]}")
    suggestions.append("Show me missing values in the dataset")
    if len(suggestions) < 4:
        suggestions.append("Describe this dataset")
    suggestions = suggestions[:4]

    for i, suggestion in enumerate(suggestions):
        with q_cols[i]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

    st.markdown("---")

    # ── Chat history display ───────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            with st.chat_message("assistant", avatar="🤖"):
                ds_name_disp = st.session_state.get("dataset_name", "your dataset")
                sheet_disp = st.session_state.get("active_sheet", "")
                sheet_suffix = f" (sheet: **{sheet_disp}**)" if sheet_disp else ""
                st.markdown(
                    f"👋 Hello! I'm your AI Data Agent. I've loaded **`{ds_name_disp}`**{sheet_suffix} "
                    f"with **{len(df):,} rows** and **{len(df.columns)} columns** "
                    f"({', '.join(f'`{c}`' for c in df.columns[:5])}"
                    f"{'...' if len(df.columns) > 5 else ''}). "
                    "Ask me anything about this data!"
                )
        else:
            for msg in st.session_state.chat_history:
                render_chat_message(
                    msg["role"],
                    msg["content"],
                    msg.get("tools_used"),
                )
                if msg.get("table_data") is not None:
                    st.dataframe(msg["table_data"], use_container_width=True)
                if msg.get("chart_data") is not None:
                    _chart_data = msg["chart_data"]
                    if isinstance(_chart_data, dict):
                        fig = px.bar(
                            x=list(_chart_data.keys()),
                            y=list(_chart_data.values()),
                            labels={"x": "Category", "y": "Value"},
                        )
                        fig.update_layout(
                            template="plotly_white",
                            margin=dict(l=20, r=20, t=30, b=20),
                            height=350,
                        )
                        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    pending = st.session_state.pop("pending_question", None)
    ds_display = st.session_state.get("dataset_name", "your dataset")
    user_input = st.chat_input(f"Ask a question about {ds_display}...")

    question = pending or user_input
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("🔍 Analysing your data..."):
            api = st.session_state.get("api_status")
            if not api or not api.get("ok"):
                check_api_status()
                api = st.session_state.get("api_status")

            if api and api.get("ok"):
                agent_result = run_agent(
                    user_message=question,
                    df=df,
                    chat_history=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history[:-1]
                        if m.get("content")
                    ],
                    client=api["client"],
                    model=api.get("model", "gemini-3.5-flash-lite"),
                )
            else:
                agent_result = run_agent_without_llm(question, df)

        final_answer = (agent_result.get("response") or "").strip()
        if not final_answer:
            fallback = run_agent_without_llm(question, df)
            final_answer = fallback.get("response") or "Analysis completed successfully."
            if not agent_result.get("table_data"):
                agent_result["table_data"] = fallback.get("table_data")
            if not agent_result.get("tools_used"):
                agent_result["tools_used"] = fallback.get("tools_used", [])

        assistant_msg = {
            "role": "assistant",
            "content": final_answer,
            "tools_used": agent_result.get("tools_used"),
        }
        if agent_result.get("table_data") is not None:
            assistant_msg["table_data"] = agent_result["table_data"]
        if agent_result.get("chart_data") is not None:
            assistant_msg["chart_data"] = agent_result["chart_data"]

        st.session_state.chat_history.append(assistant_msg)
        st.rerun()


# ── Tab: Data Preview ─────────────────────────────────────────────────────
with tab_preview:
    st.markdown("### 📋 Dataset Preview")

    # Active sheet banner
    active_sheet = st.session_state.get("active_sheet")
    sheets = st.session_state.get("available_sheets", [])
    if active_sheet and sheets:
        st.markdown(
            f'<div class="sheet-banner">📑 Viewing sheet <strong>{active_sheet}</strong>'
            f'{(" of " + str(len(sheets)) + " available sheets") if len(sheets) > 1 else ""}</div>',
            unsafe_allow_html=True,
        )

    # Column info table
    col_info = get_column_info(df)
    col_info_formatted = [
        {
            "Column": c["column"],
            "Data Type": c["dtype"],
            "Non-Null Count": c["non_null_count"],
            "Sample Values": ", ".join(map(str, c["sample_values"])),
        }
        for c in col_info
    ]
    col_info_df = pd.DataFrame(col_info_formatted)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("**Column Information**")
        st.dataframe(col_info_df, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("**Dataset Shape**")
        st.markdown(f"- **Rows:** {len(df):,}")
        st.markdown(f"- **Columns:** {len(df.columns)}")
        st.markdown(f"- **Numeric columns:** {len(numeric_cols)}")
        st.markdown(f"- **Categorical columns:** {len(cat_cols)}")

        if st.session_state.meta:
            notes = st.session_state.meta.get("cleaning_notes", [])
            if notes:
                st.markdown("**Data Cleaning Notes:**")
                for note in notes:
                    st.markdown(f"- {note}")

    st.markdown("---")
    st.markdown("### 🔎 Full Dataset")
    st.dataframe(df, use_container_width=True, hide_index=True, height=500)


# ── Tab: Analytics ─────────────────────────────────────────────────────────
with tab_analytics:
    st.markdown("### 📈 Dataset Analytics")

    if not numeric_cols:
        st.info("No numeric columns available for analytics charts.")
    else:
        # ── Row 1: Two primary charts, fully generic ──────────────────────
        ch1, ch2 = st.columns(2)

        with ch1:
            # Bar chart: top 15 values of first numeric column
            col0 = numeric_cols[0]
            # Try to use first text column as label axis
            label_col = cat_cols[0] if cat_cols else df.index.name or "index"
            st.markdown(f"**📊 Top 15 by `{col0}`**")

            if cat_cols:
                chart_df = df.nlargest(15, col0)[[cat_cols[0], col0]].copy()
                fig = px.bar(
                    chart_df,
                    x=cat_cols[0],
                    y=col0,
                    color=col0,
                    color_continuous_scale="Blues",
                )
            else:
                chart_df = df.nlargest(15, col0)[[col0]].copy()
                chart_df["_idx"] = chart_df.index.astype(str)
                fig = px.bar(chart_df, x="_idx", y=col0, color=col0, color_continuous_scale="Blues")

            fig.update_layout(
                template="plotly_white",
                margin=dict(l=20, r=20, t=20, b=60),
                height=400,
                xaxis_tickangle=-45,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            if len(numeric_cols) >= 2:
                # Grouped bar for first 2 numeric cols, top 10 by first
                col1 = numeric_cols[1]
                st.markdown(f"**📦 `{col0}` vs `{col1}` — Top 10**")
                if cat_cols:
                    top10 = df.nlargest(10, col0)
                    melted = top10.melt(
                        id_vars=[cat_cols[0]],
                        value_vars=[col0, col1],
                        var_name="Metric",
                        value_name="Value",
                    )
                    fig = px.bar(
                        melted,
                        x=cat_cols[0],
                        y="Value",
                        color="Metric",
                        barmode="group",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig.update_layout(
                        template="plotly_white",
                        margin=dict(l=20, r=20, t=20, b=60),
                        height=400,
                        xaxis_tickangle=-45,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No categorical column available for X-axis.")
            else:
                # Histogram of first numeric column
                st.markdown(f"**📦 Distribution of `{col0}`**")
                fig = px.histogram(df, x=col0, nbins=30, color_discrete_sequence=["#3b82f6"])
                fig.update_layout(template="plotly_white", height=400, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

        # ── Row 2: Pie + Scatter (or histogram fallback) ──────────────────
        ch3, ch4 = st.columns(2)

        with ch3:
            if cat_cols and numeric_cols:
                col0 = numeric_cols[0]
                top10_pie = df.nlargest(10, col0)
                st.markdown(f"**💰 Top 10 by `{col0}` (share)**")
                fig = px.pie(
                    top10_pie,
                    values=col0,
                    names=cat_cols[0],
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.4,
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=400,
                    legend=dict(font=dict(size=10)),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Box plot of all numeric columns
                st.markdown("**💰 Numeric Column Box Plot**")
                melted_box = df[numeric_cols].melt(var_name="Column", value_name="Value")
                fig = px.box(melted_box, x="Column", y="Value", color="Column",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(template="plotly_white", height=400, margin=dict(l=20, r=20, t=20, b=60),
                                  xaxis_tickangle=-30, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with ch4:
            if len(numeric_cols) >= 2:
                col0, col1 = numeric_cols[0], numeric_cols[1]
                st.markdown(f"**📉 `{col0}` vs `{col1}` — Scatter**")
                scatter_kwargs = dict(x=col0, y=col1)
                if cat_cols:
                    scatter_kwargs["text"] = cat_cols[0]
                if len(numeric_cols) >= 3:
                    scatter_kwargs["size"] = numeric_cols[2]
                    scatter_kwargs["color"] = numeric_cols[2]
                    scatter_kwargs["color_continuous_scale"] = "Viridis"
                fig = px.scatter(df, **scatter_kwargs)
                if cat_cols:
                    fig.update_traces(textposition="top center", textfont_size=8)
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Just histogram of first numeric column
                st.markdown(f"**📉 Histogram of `{numeric_cols[0]}`**")
                fig = px.histogram(df, x=numeric_cols[0], nbins=30,
                                   color_discrete_sequence=["#8b5cf6"])
                fig.update_layout(template="plotly_white", height=400, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

        # ── Descriptive stats ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Descriptive Statistics")
        desc = df[numeric_cols].describe()
        st.dataframe(desc, use_container_width=True)

        # ── Correlation matrix ────────────────────────────────────────────
        if len(numeric_cols) > 1:
            st.markdown("### 🔗 Correlation Matrix")
            corr = df[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                aspect="auto",
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)
