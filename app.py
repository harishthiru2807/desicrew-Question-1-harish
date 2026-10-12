"""
app.py — DesiCrew AI Data Agent
Streamlit web application for interactive inventory data analysis.

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

from src.data_loader import load_inventory_excel, get_dataset_summary, get_column_info
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

    /* ─── Hide Streamlit branding ─── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "meta" not in st.session_state:
    st.session_state.meta = None
if "api_status" not in st.session_state:
    st.session_state.api_status = None


# ── Helper functions ───────────────────────────────────────────────────────
def auto_load_default_dataset():
    """Attempt to load the bundled dataset automatically."""
    default_path = Path(__file__).parent / "Inventory-Records-Sample-Data.xlsx"
    if default_path.exists() and st.session_state.df is None:
        try:
            df, meta = load_inventory_excel(str(default_path))
            st.session_state.df = df
            st.session_state.meta = meta
            return True
        except Exception:
            return False
    return st.session_state.df is not None


def check_api_status():
    """Check if the OpenAI API is configured and reachable."""
    client, result = get_api_client()
    if client is None:
        st.session_state.api_status = {"ok": False, "message": result}
        return False
    st.session_state.api_status = {"ok": True, "client": client, "model": result}
    return True


def render_chat_message(role: str, content: str, tools_used=None):
    """Render a chat message with appropriate styling."""
    if role == "user":
        st.markdown(f'<div class="chat-user">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-assistant">{content}</div>', unsafe_allow_html=True)
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
            '<div class="status-ok">🟢 AI Agent Active (DesiCrew Gemini)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-ok">🟢 AI Agent Active</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Dataset status
    auto_load_default_dataset()
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(
            f'<div class="status-ok">📁 Dataset Loaded — {len(df)} products</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-error">📁 No Dataset Loaded</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # File upload
    st.markdown("### 📤 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload an Excel file (.xlsx)",
        type=["xlsx", "xls"],
        help="Upload your inventory Excel file for analysis",
    )
    if uploaded_file is not None:
        try:
            df, meta = load_inventory_excel(uploaded_file)
            st.session_state.df = df
            st.session_state.meta = meta
            st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    st.markdown("---")

    # Chat controls
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem; color:#64748b; text-align:center;'>"
        "DesiCrew Solutions<br>Data Science Internship — Q1<br>"
        "AI Inventory Data Agent</p>",
        unsafe_allow_html=True,
    )


# ── Main content area ─────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 AI Inventory Data Agent</h1>
    <p>Ask questions about your inventory data in plain English. Powered by AI with safe, validated data operations.</p>
</div>
""", unsafe_allow_html=True)


# If no dataset loaded, show upload prompt
if st.session_state.df is None:
    st.warning("⬆️ **Please upload an Excel dataset** using the sidebar to get started.")
    st.info(
        "This agent analyses inventory data. Upload an Excel file with product IDs, "
        "names, stock levels, costs, and other inventory metrics."
    )
    st.stop()

df = st.session_state.df

# ── Summary metrics row ───────────────────────────────────────────────────
summary = get_dataset_summary(df)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Total Products", f"{summary['total_products']}")
with m2:
    val = summary.get("total_inventory_value", 0)
    st.metric("Inventory Value", f"${val:,.0f}")
with m3:
    avg = summary.get("avg_cost_per_unit", 0)
    st.metric("Avg Unit Cost", f"${avg:,.2f}")
with m4:
    if "Number of Units Sold" in df.columns:
        sold = int(df["Number of Units Sold"].sum())
        st.metric("Total Units Sold", f"{sold:,}")
    else:
        st.metric("Columns", f"{len(df.columns)}")
with m5:
    st.metric("Missing Values", f"{summary['total_missing']}")

st.markdown("")

# ── Tabs: Chat | Data Preview | Analytics ─────────────────────────────────
tab_chat, tab_preview, tab_analytics = st.tabs(["💬 Chat", "📋 Data Preview", "📈 Analytics"])

# ── Tab: Chat ──────────────────────────────────────────────────────────────
with tab_chat:
    # Suggested questions
    st.markdown("**💡 Try asking:**")
    q_cols = st.columns(4)
    suggestions = [
        "What are the 5 most expensive products?",
        "How many products have stock below 30?",
        "What is the total inventory value?",
        "Show me missing values in the dataset",
    ]
    for i, suggestion in enumerate(suggestions):
        with q_cols[i]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

    st.markdown("---")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                '<div class="chat-assistant">'
                '👋 Hello! I\'m your AI Inventory Data Agent. I can help you analyze your '
                'inventory dataset. Ask me anything about your products, stock levels, costs, '
                'or any other data insights you need!</div>',
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.chat_history:
                render_chat_message(
                    msg["role"],
                    msg["content"],
                    msg.get("tools_used"),
                )
                # Show table if present
                if msg.get("table_data") is not None:
                    st.dataframe(msg["table_data"], use_container_width=True)
                # Show chart if present
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

    # Chat input
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask a question about your inventory data...")

    question = pending or user_input
    if question:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        # Process the question
        with st.spinner("🔍 Analyzing your data..."):
            api = st.session_state.api_status
            if api and api["ok"]:
                # Use LLM agent
                agent_result = run_agent(
                    user_message=question,
                    df=df,
                    chat_history=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history[:-1]
                    ],
                    client=api["client"],
                    model=api["model"],
                )
            else:
                # Use fallback agent
                agent_result = run_agent_without_llm(question, df)

        # Add assistant response to history
        assistant_msg = {
            "role": "assistant",
            "content": agent_result["response"],
            "tools_used": agent_result.get("tools_used"),
        }

        # Attach table data
        if agent_result.get("table_data") is not None:
            assistant_msg["table_data"] = agent_result["table_data"]

        # Attach chart data
        if agent_result.get("chart_data") is not None:
            assistant_msg["chart_data"] = agent_result["chart_data"]

        st.session_state.chat_history.append(assistant_msg)
        st.rerun()


# ── Tab: Data Preview ─────────────────────────────────────────────────────
with tab_preview:
    st.markdown("### 📋 Dataset Preview")

    # Column info
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
        st.markdown(f"- **Rows:** {len(df)}")
        st.markdown(f"- **Columns:** {len(df.columns)}")

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
    st.markdown("### 📈 Inventory Analytics")

    # Check available numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns available for analytics.")
    else:
        # ── Row 1: Charts ──
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("**📊 Cost Price Per Unit Distribution**")
            if "Cost Price Per Unit (USD)" in df.columns and "Product Name" in df.columns:
                chart_df = df.sort_values("Cost Price Per Unit (USD)", ascending=False).head(15)
                fig = px.bar(
                    chart_df,
                    x="Product Name",
                    y="Cost Price Per Unit (USD)",
                    color="Cost Price Per Unit (USD)",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=20, b=60),
                    height=400,
                    xaxis_tickangle=-45,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Required columns not found.")

        with ch2:
            st.markdown("**📦 Stock Levels Comparison**")
            stock_cols = ["Opening Stock", "Purchase/Stock In", "Number of Units Sold", "Hand-In-Stock"]
            available_stock = [c for c in stock_cols if c in df.columns]
            if available_stock and "Product Name" in df.columns:
                top_products = df.nlargest(10, "Hand-In-Stock") if "Hand-In-Stock" in df.columns else df.head(10)
                melted = top_products.melt(
                    id_vars=["Product Name"],
                    value_vars=available_stock,
                    var_name="Metric",
                    value_name="Units",
                )
                fig = px.bar(
                    melted,
                    x="Product Name",
                    y="Units",
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
                st.info("Stock columns not found.")

        # ── Row 2: More charts ──
        ch3, ch4 = st.columns(2)

        with ch3:
            st.markdown("**💰 Total Cost Distribution (Top 10)**")
            if "Cost Price Total (USD)" in df.columns and "Product Name" in df.columns:
                top10 = df.nlargest(10, "Cost Price Total (USD)")
                fig = px.pie(
                    top10,
                    values="Cost Price Total (USD)",
                    names="Product Name",
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
                st.info("Required columns not found.")

        with ch4:
            st.markdown("**📉 Units Sold vs Hand-In-Stock**")
            if "Number of Units Sold" in df.columns and "Hand-In-Stock" in df.columns and "Product Name" in df.columns:
                fig = px.scatter(
                    df,
                    x="Number of Units Sold",
                    y="Hand-In-Stock",
                    text="Product Name",
                    size="Cost Price Per Unit (USD)" if "Cost Price Per Unit (USD)" in df.columns else None,
                    color="Cost Price Per Unit (USD)" if "Cost Price Per Unit (USD)" in df.columns else None,
                    color_continuous_scale="Viridis",
                )
                fig.update_traces(textposition="top center", textfont_size=8)
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Required columns not found.")

        # ── Descriptive stats table ──
        st.markdown("---")
        st.markdown("### 📊 Descriptive Statistics")
        desc = df[numeric_cols].describe()
        st.dataframe(desc, use_container_width=True)

        # ── Correlation matrix ──
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
