# 📊 DesiCrew AI Inventory Data Agent

> **DesiCrew Solutions — Data Science Internship Evaluation — Question 1**

An AI-powered web application that answers natural-language questions about inventory datasets. Built with Streamlit, Pandas, and OpenAI's function-calling API.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💬 **Natural Language Chat** | Ask questions about your inventory data in plain English |
| 🔍 **Safe Data Querying** | Whitelisted operations only — no `eval`/`exec` on AI-generated code |
| 📊 **Interactive Charts** | Auto-generated visualisations with Plotly |
| 🌐 **Web Search** | Look up definitions and external context via DuckDuckGo |
| 📋 **Data Preview** | Browse the full dataset with column info and cleaning notes |
| 📈 **Analytics Dashboard** | Pre-built charts for cost distribution, stock levels, correlations |
| 🔒 **Secure** | API keys via env vars or Streamlit secrets — never hardcoded |
| 🤖 **Graceful Fallback** | Works without an API key for basic queries |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                    │
│  ┌──────────┬──────────────┬──────────────────┐  │
│  │ Sidebar  │  Chat Tab    │ Analytics Tab    │  │
│  │ • Upload │  • Messages  │ • Charts         │  │
│  │ • Config │  • Input     │ • Stats          │  │
│  └──────────┴──────────────┴──────────────────┘  │
├─────────────────────────────────────────────────┤
│              AI Agent (src/agent.py)             │
│  • System prompt with dataset context            │
│  • OpenAI function-calling                       │
│  • Tool dispatch and result formatting           │
├──────────────────┬──────────────────────────────┤
│  Safe Executor   │    Search Tool               │
│  (Whitelisted    │    (DuckDuckGo)              │
│   operations)    │                               │
├──────────────────┴──────────────────────────────┤
│            Data Loader (src/data_loader.py)       │
│  • Excel parsing with header detection           │
│  • Column normalisation                          │
│  • Type coercion and cleaning                    │
└─────────────────────────────────────────────────┘
```

## 🚀 Quick Start (Windows PowerShell)

### 1. Clone or navigate to the project

```powershell
cd "e:\projects\desicrew question 1"
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure your AI Provider (Free options available!)

The app supports multiple providers. **Groq is 100% free forever** with no credit card required:

**Option A — Groq (Recommended & 100% Free):**
1. Get a free API key at [console.groq.com/keys](https://console.groq.com/keys) (takes 30 seconds)
2. In PowerShell:
```powershell
$env:GROQ_API_KEY = "gsk_your-actual-key-here"
```

**Option B — Google Gemini (Free tier):**
1. Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. In PowerShell:
```powershell
$env:GEMINI_API_KEY = "AIzaSy_your-actual-key-here"
```

**Option C — OpenAI:**
```powershell
$env:OPENAI_API_KEY = "sk-your-actual-key-here"
```

**Option D — Enter key directly in the web UI sidebar:**
Select your provider in the sidebar dropdown and paste your key.

**Option E — 100% Offline Mode (No key needed):**
If no key is configured or if your API credits run out, the app automatically runs in **offline mode**, answering inventory questions, calculating statistics, detecting discrepancies, and displaying charts locally!

### 5. Run the application

```powershell
streamlit run app.py
```

The app opens at **http://localhost:8501**.

### 6. Run automated tests

```powershell
pytest tests/ -v
```

## 📁 Project Structure

```
desicrew question 1/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── .env.example                        # Configuration template
├── .gitignore                          # Git ignore rules
├── README.md                           # Documentation
├── Inventory-Records-Sample-Data.xlsx  # Supplied dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Excel parsing and cleaning
│   ├── safe_executor.py                # Whitelisted safe data operations
│   ├── search_tool.py                  # DuckDuckGo web search
│   └── agent.py                        # Multi-provider agent & offline engine
└── tests/
    ├── test_data_loader.py             # Data loader & safe executor tests
    └── test_agent.py                   # Multi-provider agent & quota tests
```

## 🔐 Security

- **No hardcoded API keys** — keys are loaded from environment variables, `.env` files, or Streamlit secrets.
- **No `eval`/`exec`** — the agent uses a whitelist of safe, parameterised operations instead of executing arbitrary AI-generated code.
- **Input validation** — all tool parameters are validated before execution.
- **Graceful error handling** — invalid questions, missing data, and API failures are handled with clear messages.

## 🤖 Supported Operations

The agent can perform these safe data operations:

| Operation | Description |
|-----------|-------------|
| `describe` | Descriptive statistics |
| `head` / `tail` | First / last N rows |
| `shape` / `columns` / `dtypes` | Dataset structure info |
| `missing_values` | Missing value analysis |
| `filter` | Filter by column conditions (==, !=, >, <, contains, etc.) |
| `sort` | Sort by columns |
| `top_n` / `bottom_n` | Top / bottom N by a column |
| `aggregate` | Sum, mean, median, min, max, count, std, var, nunique |
| `group_by` | Group and aggregate |
| `value_counts` / `unique_values` | Frequency analysis |
| `correlation` | Correlation matrix |
| `select_columns` | Column selection |
| `search_rows` | Text search across all columns |
| `calculate_column` | Safe arithmetic between columns |
| `percentile` / `crosstab` | Statistical operations |

## 🌐 Deployment (Streamlit Community Cloud)

1. Push the project to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository and select `app.py` as the main file.
4. Add your `OPENAI_API_KEY` in **Settings → Secrets**.
5. Deploy!

## 📝 Dataset

The supplied dataset (`Inventory-Records-Sample-Data.xlsx`) contains **46 products** (P101–P146) with 8 columns:

| Column | Type | Description |
|--------|------|-------------|
| Product ID | String | Unique product identifier (P101–P146) |
| Product Name | String | Product name |
| Opening Stock | Integer | Initial stock quantity |
| Purchase/Stock In | Integer | Units purchased/received |
| Number of Units Sold | Integer | Units sold |
| Hand-In-Stock | Integer | Current stock (= Opening + Purchase - Sold) |
| Cost Price Per Unit (USD) | Float | Price per unit in USD |
| Cost Price Total (USD) | Float | Total cost (= Unit Price × Hand-In-Stock) |

---

**Built for the DesiCrew Solutions Data Science Internship Evaluation.**
