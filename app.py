"""
app.py  ─  SENTINEL: AI Crime & Safety Analytics System
============================================================
Stack:  YOLOv8 · LangChain · Random Forest · Groq LLM · Plotly · Streamlit

Run:
    streamlit run app.py
"""
import io
import os
import sys
import warnings
import tempfile
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# HELPER — SAMPLE IMAGE GENERATOR
# ══════════════════════════════════════════════════════════════════════════
def _make_sample_image(scene: str = "knife") -> Image.Image:
    """Create a synthetic CCTV-style scene for demo detection."""
    rng = random.Random(42)
    W, H = 640, 480
    img = Image.new("RGB", (W, H), color=(8, 11, 18))

    try:
        draw = ImageDraw.Draw(img)
        # Ground plane
        draw.rectangle([(0, H*2//3), (W, H)], fill=(20, 25, 40))
        # Background structures
        for bx in [80, 260, 420, 560]:
            bh = rng.randint(120, 220)
            draw.rectangle([(bx, H*2//3 - bh), (bx+80, H*2//3)], fill=(25, 32, 52))
        # Person silhouettes
        for px, py in [(120, 240), (380, 260)] if scene != "clear" else [(200, 260)]:
            draw.ellipse([(px-18, py-50), (px+18, py)],    fill=(80, 90, 120))
            draw.rectangle([(px-22, py), (px+22, py+80)],  fill=(60, 70, 100))
            draw.rectangle([(px-20, py+80), (px-6, py+130)], fill=(50, 60, 90))
            draw.rectangle([(px+6,  py+80), (px+20, py+130)], fill=(50, 60, 90))
        # Knife overlay
        if scene == "knife":
            kx, ky = 148, 310
            draw.polygon([(kx,ky),(kx+4,ky+40),(kx-4,ky+40)], fill=(220, 230, 240))
            draw.line([(kx,ky+40),(kx,ky+60)], fill=(140,120,100), width=3)
        # Scanline
        for y in range(0, H, 4):
            draw.line([(0,y),(W,y)], fill=(0,200,232,5), width=1)
        # Timestamp overlay
        draw.text((8, 8),  "CAM_04 · SECTOR B", fill=(0, 200, 232))
        draw.text((8, 24), "2024-03-15 14:28:44", fill=(148, 163, 184))
        draw.text((W-80, 8), "● REC",  fill=(230, 57, 70))
    except Exception:
        pass

    return img

# ── local modules ──
sys.path.insert(0, str(Path(__file__).parent))
from modules.data_loader    import (load_or_generate, engineer_features,
                                    setup_kaggle, download_kaggle_dataset,
                                    generate_sample_data, FEATURE_COLS, TARGET_COL,
                                    CRIME_TYPES)
from modules.ml_model       import (train_random_forest, load_model, forecast_zone_risk,
                                    fig_feature_importance, fig_confusion_matrix,
                                    fig_forecast, fig_class_distribution, fig_cv_scores,
                                    LABEL_MAP, SEV_COLORS)
from modules.vision         import (run_detection, get_model, extract_key_frames,
                                    fig_detection_summary, fig_confidence_dist)
from modules.visualizations import (fig_density_heatmap, fig_scatter_map,
                                    fig_hourly_distribution, fig_weekly_trend,
                                    fig_crime_type_bar, fig_district_breakdown,
                                    fig_arrest_rate, fig_dayofweek_heatmap,
                                    fig_monthly_trend)
from modules.report_gen     import (summarize_crime_data, generate_report_groq)

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SENTINEL — AI Crime Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# DARK THEME  CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ─── Global & Modern Scrollbar ─── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Outfit:wght@300;400;600&display=swap');

html, body, [class*="css"]  { font-family: 'Outfit', sans-serif; }
.main {
    background: radial-gradient(circle at top right, hsla(220, 40%, 12%, 1), hsla(220, 40%, 5%, 1));
    background-attachment: fixed;
}
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: hsla(190, 100%, 50%, 0.1); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: hsla(190, 100%, 50%, 0.3); }

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background-color: hsla(225, 45%, 7%, 0.95);
    backdrop-filter: blur(16px);
    border-right: 1px solid hsla(220, 30%, 20%, 0.5);
}

/* ─── Metric cards (Premium Glass) ─── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, hsla(225, 35%, 15%, 0.4), hsla(225, 35%, 10%, 0.5));
    backdrop-filter: blur(12px);
    border: 1px solid hsla(220, 30%, 25%, 0.3);
    border-radius: 16px;
    padding: 1.2rem !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 8px 32px hsla(0, 0%, 0%, 0.4);
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    border-color: hsla(190, 100%, 50%, 0.5);
    box-shadow: 0 12px 40px hsla(190, 100%, 50%, 0.15);
}
[data-testid="stMetricLabel"] {
    color: hsla(215, 20%, 75%, 1) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
}
[data-testid="stMetricLabel"] > div {
    overflow: visible !important;
    white-space: normal !important;
    line-height: 1.3;
}
[data-testid="stMetricValue"] {
    color: hsla(190, 100%, 75%, 1) !important;
    font-size: 1.8rem !important;
    font-weight: 600;
    font-family: 'Share Tech Mono', monospace;
}

/* ─── Tabs ─── */
[data-testid="stTabs"] { background: transparent; }
[data-testid="stTabs"] button {
    color: hsla(215, 20%, 60%, 1);
    font-size: 0.9rem;
    font-weight: 600;
    border-bottom: 2px solid transparent !important;
    transition: all 0.3s;
    padding: 0.5rem 1.5rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: hsla(190, 100%, 50%, 1) !important;
    border-bottom-color: hsla(190, 100%, 50%, 1) !important;
    background: hsla(190, 100%, 50%, 0.08);
}

/* ─── Buttons ─── */
.stButton>button {
    background: hsla(225, 30%, 15%, 0.7);
    color: hsla(190, 100%, 70%, 1);
    border: 1px solid hsla(220, 30%, 30%, 0.6);
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    transition: all 0.3s ease;
    backdrop-filter: blur(4px);
}
.stButton>button:hover {
    background: hsla(190, 100%, 50%, 0.15);
    border-color: hsla(190, 100%, 50%, 0.7);
    color: white;
    box-shadow: 0 0 20px hsla(190, 100%, 50%, 0.2);
}

/* ─── Dataframe & Tables ─── */
div[data-testid="stExpander"] {
    background: hsla(225, 30%, 10%, 0.4);
    border: 1px solid hsla(220, 30%, 20%, 0.5);
    border-radius: 10px;
}

/* ─── Alerts ─── */
.alert-critical { background:hsla(355, 75%, 10%, 0.7); border:1px solid hsla(355, 75%, 50%, 0.4); border-left:4px solid hsla(355, 75%, 50%, 1); padding:.8rem; border-radius:8px; color:hsla(355, 75%, 80%, 1); margin:.8rem 0; font-size:0.85rem; backdrop-filter: blur(4px); }
.alert-high     { background:hsla(30, 75%, 10%, 0.7); border:1px solid hsla(30, 75%, 50%, 0.4); border-left:4px solid hsla(30, 75%, 50%, 1); padding:.8rem; border-radius:8px; color:hsla(30, 75%, 80%, 1); margin:.8rem 0; font-size:0.85rem; backdrop-filter: blur(4px); }
.alert-medium   { background:hsla(140, 75%, 10%, 0.7); border:1px solid hsla(140, 75%, 50%, 0.4); border-left:4px solid hsla(140, 75%, 50%, 1); padding:.8rem; border-radius:8px; color:hsla(140, 75%, 80%, 1); margin:.8rem 0; font-size:0.85rem; backdrop-filter: blur(4px); }

/* ─── Report ─── */
.sentinel-report {
    background: hsla(225, 40%, 6%, 0.85);
    border: 1px solid hsla(190, 100%, 50%, 0.2);
    border-radius: 16px;
    padding: 2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    line-height: 1.7;
    color: hsla(210, 20%, 90%, 1);
    box-shadow: 0 10px 40px hsla(0, 0%, 0%, 0.6);
}
.sentinel-header {
    background: linear-gradient(90deg, hsla(190, 100%, 50%, 0.15), transparent);
    border-left: 5px solid hsla(190, 100%, 50%, 1);
    padding: 1.8rem;
    border-radius: 4px 16px 16px 4px;
    margin-bottom: 2.5rem;
    box-shadow: 0 4px 20px hsla(190, 100%, 50%, 0.05);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = dict(
        df=None, ml_metrics=None, yolo_model=None,
        detections=[], annotated_img=None,
        data_loaded=False, model_trained=False,
        groq_key="", kaggle_user="", kaggle_key="",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init_state()


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ SENTINEL")
    st.markdown("<p style='color:#e63946;font-size:.7rem;letter-spacing:.15em;'>AI CRIME ANALYTICS SYSTEM</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Kaggle credentials ──
    st.markdown("### 🔑 Kaggle API")
    k_user = st.text_input("Kaggle Username", placeholder="your_username", key="ki_user")
    k_key  = st.text_input("Kaggle API Key",  placeholder="xxxxxxxxxxxx",  type="password", key="ki_key")
    if st.button("⬇ Download Chicago Crime Dataset", use_container_width=True):
        if k_user and k_key:
            setup_kaggle(k_key, k_user)
            with st.spinner("Connecting to Kaggle…"):
                csv_path = download_kaggle_dataset()
            if csv_path:
                st.success(f"Downloaded: {csv_path.name}")
                st.session_state["kaggle_csv"] = str(csv_path)
            else:
                st.error("Download failed — check credentials or dataset slug.")
        else:
            st.warning("Enter Kaggle credentials first.")

    st.markdown("---")

    # ── Or upload own CSV ──
    st.markdown("### 📁 Upload Dataset")
    uploaded_csv = st.file_uploader("Crime CSV / Kaggle Export", type=["csv"])
    if uploaded_csv:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            f.write(uploaded_csv.read())
            st.session_state["upload_csv"] = f.name
        st.success(f"Uploaded: {uploaded_csv.name}")

    if st.button("⚡ Use Built-in Sample Data", use_container_width=True):
        st.session_state["use_sample"] = True

    st.markdown("---")

    # ── Model params ──
    st.markdown("### ⚙️ Model Parameters")
    n_trees   = st.slider("RF Trees",         50, 500, 250, step=50)
    max_depth = st.slider("Max Depth",          4,  30,  18, step=2)
    test_split= st.slider("Test Split %",       10,  40,  20, step=5)
    conf_thr  = st.slider("YOLO Confidence %",  20,  90,  40, step=5)
    n_sample  = st.selectbox("Sample Size", [5_000, 10_000, 15_000, 25_000], index=2)

    st.markdown("---")

    # ── API keys ──
    st.markdown("### 🤖 AI Keys (optional)")
    groq_key = st.text_input("Groq API Key", placeholder="gsk_...", type="password")
    if groq_key:
        st.session_state["groq_key"] = groq_key

    st.markdown("---")
    st.markdown("<p style='color:#4a5568;font-size:.65rem;'>YOLOv8 · LangChain · Random Forest<br>Groq LLM · Plotly · Streamlit</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
    <div class='sentinel-header'>
        <h1>SENTINEL · ANALYTICS</h1>
        <p>Advanced Crime Prediction & Weapon Detection Intelligence</p>
    </div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
df_now = st.session_state.get("df")
n_recs   = len(df_now) if df_now is not None else 0
trained  = st.session_state.get("model_trained", False)
metrics  = st.session_state.get("ml_metrics") or {}
dets     = st.session_state.get("detections", [])

c1.metric("📊 Dataset Records",  f"{n_recs:,}",      delta="Loaded" if n_recs else "—")
c2.metric("🧠 Model Accuracy",   f"{metrics.get('accuracy',0):.1%}" if trained else "—",
          delta=f"F1 {metrics.get('f1_score',0):.3f}" if trained else None)
c3.metric("🎯 AUC-ROC",          f"{metrics.get('auc_roc',0):.3f}"  if trained else "—")
c4.metric("👁️ YOLO Detections",  str(len(dets)),     delta="Active" if dets else "—")
c5.metric("⚠️ Weapons Found",
          str(sum(1 for d in dets if d.get("category") == "WEAPON")),
          delta="CRITICAL" if any(d.get("severity")=="CRITICAL" for d in dets) else "CLEAR",
          delta_color="inverse" if any(d.get("severity")=="CRITICAL" for d in dets) else "normal")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📥 Data",
    "👁 Vision",
    "🗺️ Heatmap",
    "🧠 Predict",
    "📊 Trends",
    "📋 AI Report",
])

# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA
# ──────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### 1 · Load Dataset & Train Random Forest")
    col_load, col_train = st.columns([1, 1])

    with col_load:
        st.markdown("#### Load Data")
        st.info("📌 Upload a CSV above, download the Kaggle Chicago Crime dataset, or use built-in sample data.")

        csv_path = (st.session_state.get("upload_csv")
                    or st.session_state.get("kaggle_csv"))
        use_sample = st.session_state.get("use_sample", False)

        if st.button("🔄 Load / Refresh Dataset", use_container_width=True):
            with st.spinner("Loading and engineering features…"):
                if use_sample:
                    df = load_or_generate(n_sample=n_sample)
                    st.info(f"Generated {n_sample:,} synthetic Chicago-style records.")
                else:
                    df = load_or_generate(csv_path, n_sample=n_sample)
                    if csv_path:
                        st.success(f"Loaded CSV: {Path(csv_path).name}")
                    else:
                        st.info("No CSV found — using built-in sample data.")
                
                if df is not None and not df.empty:
                    st.session_state["df"] = df
                    st.session_state["data_loaded"] = True
                    st.rerun()
                else:
                    st.error("Failed to load dataset or dataset is empty.")

        if st.session_state["data_loaded"] and df_now is not None:
            st.markdown("##### Dataset Preview")
            st.dataframe(df_now[[
                "Date", "Primary Type", "Location Description",
                "District", "Latitude", "Longitude", "Severity"
            ]].head(200), width="stretch", height=260)

            st.markdown("##### Quick Stats")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Records",       f"{len(df_now):,}")
            q2.metric("Crime Types",   df_now["Primary Type"].nunique())
            q3.metric("Districts",     df_now["District"].nunique())
            q4.metric("Date Span",     f"{(df_now['Date'].max()-df_now['Date'].min()).days} days")

            # NLP summary
            with st.expander("🔤 LangChain NLP Summary", expanded=False):
                if st.button("Generate NLP Summary"):
                    with st.spinner("LangChain summarizing dataset…"):
                        summary = summarize_crime_data(df_now, st.session_state.get("groq_key",""))
                    st.markdown(f"> {summary}")

    with col_train:
        st.markdown("#### Train Random Forest")
        st.markdown(f"""
- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Trees:** `{n_trees}` &nbsp;|&nbsp; **Max Depth:** `{max_depth}`
- **Test Split:** `{test_split}%` &nbsp;|&nbsp; **Features:** `{len(FEATURE_COLS)}`
- **Target:** Crime Severity (0=Low · 1=Medium · 2=High · 3=Critical)
        """)

        if st.button("🚀 Train Model", width="stretch",
                     disabled=(df_now is None)):
            prog = st.progress(0.0, text="Initialising…")
            stat = st.empty()

            def cb(p, msg):
                prog.progress(p, text=msg)
                stat.caption(msg)

            results = train_random_forest(
                df_now, FEATURE_COLS, TARGET_COL,
                n_estimators=n_trees,
                max_depth=max_depth,
                test_size=test_split / 100,
                progress_cb=cb,
            )
            st.session_state["ml_metrics"] = results
            st.session_state["model_trained"] = True
            st.success(f"✓ Model trained — Accuracy {results['accuracy']:.1%}  |  F1 {results['f1_score']:.3f}  |  AUC {results['auc_roc']:.3f}")
            st.rerun()

        if st.session_state["model_trained"]:
            m = st.session_state["ml_metrics"]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Accuracy",   f"{m['accuracy']:.1%}")
            r2.metric("F1 Score",   f"{m['f1_score']:.3f}")
            r3.metric("AUC-ROC",    f"{m['auc_roc']:.3f}")
            r4.metric("CV Mean",    f"{m['cv_mean']:.3f} ±{m['cv_std']:.3f}")

            st.markdown("##### Training Info")
            st.caption(f"Train samples: {m['n_train']:,}  |  Test samples: {m['n_test']:,}  |  Trees: {n_trees}")

            # Class report
            with st.expander("📄 Full Classification Report"):
                rpt = m["report"]
                rows = []
                for cls_name, vals in rpt.items():
                    if isinstance(vals, dict):
                        rows.append({"Class": cls_name, **{k: round(v,3) for k,v in vals.items()}})
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch")

            st.plotly_chart(fig_class_distribution(df_now), width="stretch")

    # ── Confusion matrix + feature importance ──
    if st.session_state["model_trained"]:
        m = st.session_state["ml_metrics"]
        st.markdown("---")
        v1, v2, v3 = st.columns(3)
        with v1:
            st.plotly_chart(fig_confusion_matrix(m["conf_matrix"], m["classes"]),
                            width="stretch")
        with v2:
            st.plotly_chart(fig_feature_importance(m["feat_importance"]),
                            width="stretch")
        with v3:
            st.plotly_chart(fig_cv_scores(m["cv_mean"], m["cv_std"]),
                            width="stretch")


# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — COMPUTER VISION
# ──────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 2 · YOLOv8 Computer Vision — Weapon & Threat Detection")
    cv_l, cv_r = st.columns([1, 1])

    with cv_l:
        st.markdown("#### Upload Image / Video Frame")
        media = st.file_uploader("Upload CCTV Image (JPG/PNG) or Video (MP4/AVI)",
                                 type=["jpg","jpeg","png","mp4","avi","mov"])

        st.markdown("**Or use a sample scene:**")
        sample_cols = st.columns(3)
        use_sample_img = sample_cols[0].button("🔪 Sample Knife Scene")
        use_crowd_img  = sample_cols[1].button("👥 Crowd Scene")
        use_clear_img  = sample_cols[2].button("🟢 Clear Scene")

        if st.button("🔍 Run YOLOv8 Detection", width="stretch"):
            with st.spinner("Loading YOLOv8n model & running inference…"):
                # Load / cache model
                if st.session_state["yolo_model"] is None:
                    st.session_state["yolo_model"] = get_model()
                model = st.session_state["yolo_model"]

                if media:
                    suffix = Path(media.name).suffix.lower()
                    if suffix in [".mp4", ".avi", ".mov"]:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(media.read())
                            tmp_path = tmp.name
                        frames = extract_key_frames(tmp_path, n_frames=6)
                        if frames:
                            img = Image.fromarray(frames[len(frames)//2])
                        else:
                            img = _make_sample_image("video")
                    else:
                        img = Image.open(io.BytesIO(media.read())).convert("RGB")
                else:
                    # Generate synthetic test scene
                    img = _make_sample_image(
                        "knife" if use_sample_img else
                        "crowd" if use_crowd_img  else
                        "clear"
                    ) if (use_sample_img or use_crowd_img or use_clear_img) else \
                        _make_sample_image("knife")

                result = run_detection(img, conf_thresh=conf_thr/100, model=model)
                st.session_state["annotated_img"] = result["image"]
                st.session_state["detections"]    = result["detections"]
                st.rerun()

    with cv_r:
        st.markdown("#### Live Detection Feed")
        ann = st.session_state.get("annotated_img")
        if ann:
            st.image(ann, caption="YOLOv8 Annotated Frame", width="stretch")
        else:
            st.markdown("""
            <div style='background: hsla(225, 35%, 10%, 0.6); border: 1px dashed hsla(190, 100%, 50%, 0.2); border-radius: 12px;
            padding: 80px 20px; text-align: center; color: hsla(215, 20%, 50%, 1); font-size: 0.9rem;
            backdrop-filter: blur(4px); box-shadow: inset 0 0 20px hsla(0, 0%, 0%, 0.2);'>
            <div style='font-size: 2.5rem; margin-bottom: 1rem;'>📹</div>
            Upload an image / video and click <b>Run Detection</b><br>
            <span style='font-size: 0.8rem; opacity: 0.7;'>or select a sample scene below</span>
            </div>""", unsafe_allow_html=True)

        dets = st.session_state.get("detections", [])
        if dets:
            st.markdown("#### Detections")
            for d in dets:
                lvl  = d["severity"]
                cls_ = "alert-critical" if lvl=="CRITICAL" else "alert-high" if lvl=="HIGH" else "alert-medium"
                st.markdown(
                    f"<div class='{cls_}'>"
                    f"<b>[{lvl}]</b> {d['name'].upper()} "
                    f"— {d['confidence']:.1f}% confidence "
                    f"| {d['category']}"
                    f"</div>", unsafe_allow_html=True)

    if st.session_state.get("detections"):
        st.markdown("---")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.plotly_chart(fig_detection_summary(st.session_state["detections"]),
                            width="stretch")
        with dc2:
            st.plotly_chart(fig_confidence_dist(st.session_state["detections"]),
                            width="stretch")


# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — CRIME HEATMAP
# ──────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 3 · Crime Density Heatmap & Hotspot Analysis")
    if df_now is None:
        st.warning("⚠️ Load a dataset first (Tab 1).")
    else:
        map_type = st.radio("Map Type", ["Density Heatmap", "Scatter (by Crime Type)"],
                            horizontal=True)
        filter_type = st.multiselect("Filter Crime Types",
                                     sorted(df_now["Primary Type"].dropna().unique()),
                                     default=None)
        df_map = df_now if not filter_type else df_now[df_now["Primary Type"].isin(filter_type)]
        df_map = df_map.dropna(subset=["Latitude","Longitude"])

        if map_type == "Density Heatmap":
            st.plotly_chart(fig_density_heatmap(df_map), width="stretch")
        else:
            st.plotly_chart(fig_scatter_map(df_map), width="stretch")

        # Hotspot table
        st.markdown("#### Top Incident Grid Cells")
        grid = (df_map.groupby(["LatBin","LonBin"])
                      .agg(incidents=("ID","count"),
                           lat_center=("Latitude","mean"),
                           lon_center=("Longitude","mean"))
                      .reset_index()
                      .sort_values("incidents", ascending=False)
                      .head(10)
                      .reset_index(drop=True))
        grid.index += 1
        grid["lat_center"] = grid["lat_center"].round(4)
        grid["lon_center"] = grid["lon_center"].round(4)
        st.dataframe(grid[["incidents","lat_center","lon_center"]],
                     width="stretch")


# ──────────────────────────────────────────────────────────────────────────
# TAB 4 — ML PREDICTION
# ──────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 4 · Random Forest Risk Prediction & 72-Hour Forecast")
    if not st.session_state["model_trained"]:
        st.warning("⚠️ Train the model first (Tab 1).")
    else:
        m = st.session_state["ml_metrics"]
        forecast_col, zones_col = st.columns([1.3, 0.7])

        with forecast_col:
            st.markdown("#### 72-Hour Risk Forecast")
            fc1, fc2 = st.columns(2)
            sel_lat = fc1.number_input("Latitude",  value=41.8827, format="%.4f")
            sel_lon = fc2.number_input("Longitude", value=-87.6233, format="%.4f")
            zone_name = st.text_input("Zone Name", "Downtown Core")

            if st.button("🔮 Run Forecast", use_container_width=True):
                mdict = {"model": m["model"], "features": m["features"],
                         "classes": m["classes"]}
                fc_df = forecast_zone_risk(mdict, sel_lat, sel_lon, hours=72)
                st.plotly_chart(fig_forecast(fc_df, zone_name), use_container_width=True)

                peak = fc_df.loc[fc_df["risk_score"].idxmax()]
                st.markdown(
                    f"<div class='alert-critical'>🔴 Peak risk: <b>{peak['datetime'].strftime('%Y-%m-%d %H:00')}</b> "
                    f"— Score {peak['risk_score']:.2f} ({peak['risk_label']})</div>",
                    unsafe_allow_html=True)

        with zones_col:
            st.markdown("#### Top Risk Zones (from Dataset)")
            if df_now is not None:
                zone_risk = (df_now.groupby("District")
                                   .agg(mean_sev=("Severity","mean"),
                                        incidents=("ID","count"))
                                   .reset_index()
                                   .sort_values("mean_sev", ascending=False)
                                   .head(8))
                zone_risk["risk_score"] = (zone_risk["mean_sev"] / 3 * 100).round(1)
                zone_risk["level"] = zone_risk["mean_sev"].apply(
                    lambda s: "CRITICAL" if s>2.2 else "HIGH" if s>1.5 else "MEDIUM" if s>0.8 else "LOW"
                )
                for _, row in zone_risk.iterrows():
                    lvl = row["level"]
                    cls_ = ("alert-critical" if lvl=="CRITICAL" else
                            "alert-high"     if lvl=="HIGH"     else "alert-medium")
                    st.markdown(
                        f"<div class='{cls_}'><b>District {int(row['District'])}</b> "
                        f"— Risk {row['risk_score']:.0f}%  [{lvl}]  "
                        f"({int(row['incidents'])} incidents)</div>",
                        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# TAB 5 — TREND ANALYSIS
# ──────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### 5 · Trend Analysis & Crime Pattern Analytics")
    if df_now is None:
        st.warning("⚠️ Load a dataset first (Tab 1).")
    else:
        st.plotly_chart(fig_weekly_trend(df_now),      width="stretch")
        t1, t2 = st.columns(2)
        with t1:
            st.plotly_chart(fig_hourly_distribution(df_now), width="stretch")
            st.plotly_chart(fig_arrest_rate(df_now),         width="stretch")
        with t2:
            st.plotly_chart(fig_crime_type_bar(df_now),  width="stretch")
            st.plotly_chart(fig_district_breakdown(df_now), width="stretch")
        st.plotly_chart(fig_dayofweek_heatmap(df_now),  width="stretch")
        st.plotly_chart(fig_monthly_trend(df_now),      width="stretch")


# ──────────────────────────────────────────────────────────────────────────
# TAB 6 — INTELLIGENCE REPORT
# ──────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### 6 · AI Intelligence Report — Groq LLM + LangChain")

    rc1, rc2, rc3 = st.columns(3)
    rpt_type   = rc1.selectbox("Report Type",
                               ["Full Incident Report","Executive Summary",
                                "Patrol Brief","Evidence Analysis","Risk Assessment"])
    rpt_period = rc2.selectbox("Time Period",
                               ["Last 24 Hours","Last 7 Days","Last 30 Days","YTD"])
    rpt_lang   = rc3.selectbox("Language", ["English","Formal (Police)",
                                             "Spanish","French"])

    if not st.session_state.get("groq_key"):
        st.info("ℹ️ Enter a Groq API key in the sidebar for AI-generated reports. "
                "Without a key, a structured template report is produced automatically.")

    if st.button("🤖 Generate Intelligence Report", width="stretch"):
        data = {
            "df":          df_now,
            "ml_metrics":  st.session_state.get("ml_metrics"),
            "detections":  st.session_state.get("detections", []),
            "report_type": rpt_type,
            "period":      rpt_period,
            "language":    rpt_lang,
        }
        report_placeholder = st.empty()
        full_report = ""

        with st.spinner("SENTINEL AI generating report…"):
            for chunk in generate_report_groq(data, st.session_state.get("groq_key","")):
                full_report += chunk
                # Apply simple section highlighting
                display = full_report.replace(
                    "===", "<span class='section-hdr'>===</span>"
                ).replace("CRITICAL", "<span style='color:#e63946;font-weight:bold'>CRITICAL</span>")
                report_placeholder.markdown(
                    f"<div class='sentinel-report'>{display}</div>",
                    unsafe_allow_html=True)

        st.session_state["last_report"] = full_report
        st.success("✓ Report generated.")

    # ── Export ──
    if st.session_state.get("last_report"):
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "⬇ Download as TXT",
            data=st.session_state["last_report"],
            file_name=f"SENTINEL_Report_{rpt_type.replace(' ','_')}.txt",
            mime="text/plain",
            width="stretch",
        )
        dl2.download_button(
            "⬇ Download as Markdown",
            data=st.session_state["last_report"],
            file_name=f"SENTINEL_Report_{rpt_type.replace(' ','_')}.md",
            mime="text/markdown",
            width="stretch",
        )


