# 🛡️ SENTINEL — AI Crime & Safety Analytics System

```
YOLOv8 · LangChain NLP · Random Forest · Groq LLM · Plotly Heatmaps · Streamlit
```

## Stack

| Module | Technology | Role |
|---|---|---|
| 👁️ Computer Vision | YOLOv8n (Ultralytics) | Weapon / fight detection in CCTV |
| 💬 NLP | LangChain + Groq | Dataset summarization |
| 🧠 Machine Learning | Random Forest (scikit-learn) | Crime severity prediction |
| 🤖 LLM Report | Groq `llama3-8b-8192` | Police intelligence report generation |
| 📊 Visualization | Plotly + Mapbox | Crime heatmaps & trend charts |
| 🌐 UI | Streamlit | Full dark-themed dashboard |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Load data (choose one):

**Option A — Kaggle Chicago Crime Dataset (real data)**
1. Get your API key from https://www.kaggle.com/settings → Account → API → Create Token
2. Enter `Username` + `API Key` in the sidebar
3. Click **"Download Chicago Crime Dataset"**

**Option B — Upload your own CSV**
- Any CSV with columns: `Date`, `Primary Type`, `Latitude`, `Longitude`, `District`

**Option C — Built-in Sample (no setup)**
- Click **"Use Built-in Sample Data"** in the sidebar
- Generates 15,000 realistic synthetic Chicago-style records instantly

### 4. Train the model
- Go to **Tab 1 → Data & Training**
- Click **"Load / Refresh Dataset"** then **"Train Model"**
- Results: Accuracy, F1, AUC-ROC, confusion matrix, feature importance

### 5. Explore all 6 tabs
1. **📥 Data & Training** — load Kaggle CSV, train Random Forest, see metrics
2. **👁 Computer Vision** — upload CCTV image/video for YOLOv8 weapon detection
3. **🗺️ Crime Heatmap** — Plotly density heatmap with Mapbox tiles
4. **🧠 ML Prediction** — 72-hour risk forecast per location
5. **📊 Trend Analysis** — 7 interactive charts (hourly, weekly, heatmap etc.)
6. **📋 Intelligence Report** — AI-generated formal police report via Groq

---

## Optional API Keys

| Key | Where to get | Used for |
|---|---|---|
| `GROQ_API_KEY` | console.groq.com | AI report generation (free tier available) |
| Kaggle creds | kaggle.com/settings | Real Chicago crime dataset download |

Set as env vars or enter in the sidebar.

```bash
export GROQ_API_KEY=gsk_your_key_here
streamlit run app.py
```

---

## Kaggle Dataset

**Chicago Crime Dataset** (`chicago/chicago-crime`)
- 8M+ real incidents from 2001–present
- Fields: Date, Block, IUCR, Primary Type, Description, Location, Arrest, District, Ward, Latitude, Longitude
- URL: https://www.kaggle.com/datasets/chicago/chicago-crime

The app auto-handles column mapping and feature engineering.

---

## Project Structure

```
sentinel/
├── app.py                   # Main Streamlit application
├── requirements.txt
├── README.md
├── modules/
│   ├── data_loader.py       # Kaggle download + feature engineering
│   ├── ml_model.py          # Random Forest training + evaluation
│   ├── vision.py            # YOLOv8 detection + annotation
│   ├── visualizations.py    # Plotly heatmaps & trend charts
│   └── report_gen.py        # Groq LLM + LangChain report generation
├── data/                    # Downloaded / generated datasets
└── models/                  # Saved RF model (rf_crime_model.pkl)
```
