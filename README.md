# 🛡️ SENTINEL — AI Crime & Safety Analytics System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SENTINEL** is an advanced, AI-powered intelligence dashboard designed for modern crime analytics and real-time safety monitoring. By integrating Computer Vision (YOLOv8), Machine Learning (Random Forest), and Large Language Models (Groq + LangChain), SENTINEL transforms raw crime data into actionable situational awareness.

---

## 🚀 Key Capabilities

*   **👁️ Visual Intelligence**: Real-time weapon and threat detection in CCTV streams using YOLOv8.
*   **🧠 Predictive Analytics**: 72-hour crime risk forecasting powered by Random Forest regression.
*   **📊 Spatio-Temporal Insights**: Dynamic Plotly heatmaps and trend analysis across Chicago's 25 districts.
*   **📋 AI Reporting**: Automated generation of formal police intelligence reports using Llama-3 (Groq).
*   **📥 Seamless Data Integration**: One-click download of the 8M+ record Chicago Crime dataset via Kaggle.

---

## 🛠️ Technology Stack

| Layer | Component | Technology |
|---|---|---|
| **Vision** | Object Detection | `YOLOv8n` (Ultralytics) |
| **Logic** | Language Framework | `LangChain` + `Groq LLM` |
| **Science** | Machine Learning | `scikit-learn` (Random Forest) |
| **Data** | Analysis | `Pandas` + `NumPy` |
| **Viz** | Mapping & Charts | `Plotly` + `Mapbox` |
| **Interface** | Dashboard | `Streamlit` (Custom Glassmorphism CSS) |

---

## 📸 Dashboard Preview

> [!TIP]
> *Experience the premium dark-themed interface with real-time analytics by running locally or via Streamlit Cloud.*

---

## ⚙️ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Hassandav0319/AI-CRIME-ANALYTICS-SYSTEM.git
cd AI-CRIME-ANALYTICS-SYSTEM
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
To enable real-time dataset downloads and AI report generation, provide your keys in the sidebar or via terminal:
```bash
export GROQ_API_KEY='your_gsk_key'
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

---

## 📁 Project Architecture

```text
SENTINEL/
├── app.py                # Main Dashboard Engine
├── packages.txt          # Linux System Dependencies
├── requirements.txt      # Python Dependencies
├── modules/
│   ├── vision.py         # YOLOv8 Computer Vision Pipeline
│   ├── data_loader.py    # Kaggle Integration & Feature Engineering
│   ├── ml_model.py       # ML Training & Risk Forecasting
│   ├── visualizations.py # Plotly Heatmaps & Analytics
│   └── report_gen.py     # AI Intelligence Report Generation
└── data/                 # Local Cache & Datasets
```

---

## 🤝 Contributing
Contributions are welcome! If you'd like to improve the detection models or add new analytics features, please feel free to fork the repo and submit a PR.

---

## 📜 License
Distributed under the **MIT License**. See `LICENSE` for more information.

---

<p align="center">
  <i>Developed for the Next Generation of Public Safety Intelligence.</i>
</p>
