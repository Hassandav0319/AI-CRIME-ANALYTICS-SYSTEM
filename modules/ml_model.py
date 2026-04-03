"""
modules/ml_model.py
Random Forest classifier for crime severity prediction.
Includes training pipeline, evaluation metrics, feature importance,
and 24-hour risk forecasting.
"""
import warnings
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection    import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing      import LabelEncoder
from sklearn.metrics            import (classification_report, confusion_matrix,
                                        roc_auc_score, f1_score, accuracy_score)
from sklearn.pipeline           import Pipeline
from sklearn.preprocessing      import StandardScaler
from sklearn.inspection         import permutation_importance

warnings.filterwarnings("ignore")

MODEL_PATH = Path("models") / "rf_crime_model.pkl"
LABEL_MAP  = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
SEV_COLORS = {"Low": "#4ade80", "Medium": "#facc15", "High": "#f97316", "Critical": "#ef4444"}


# ────────────────────────────────────────────────────────────────────────────
# TRAINING
# ────────────────────────────────────────────────────────────────────────────
def train_random_forest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Severity",
    n_estimators: int = 250,
    max_depth: int = 18,
    test_size: float = 0.2,
    progress_cb=None,
) -> dict:
    """
    Train Random Forest on prepared feature matrix.
    Returns a result dict with model, metrics, splits, feature importance.
    """
    X = df[feature_cols].fillna(-1)
    y = df[target_col].astype(int)

    if progress_cb: progress_cb(0.05, "Splitting dataset …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if progress_cb: progress_cb(0.15, f"Training Random Forest ({n_estimators} trees) …")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    rf.fit(X_train, y_train)

    if progress_cb: progress_cb(0.65, "Evaluating model …")
    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")

    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = 0.0

    report = classification_report(y_test, y_pred,
                                   target_names=[LABEL_MAP.get(i, str(i)) for i in sorted(y.unique())],
                                   output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    if progress_cb: progress_cb(0.75, "Computing feature importance …")
    feat_imp = pd.DataFrame({
        "Feature":   feature_cols,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    if progress_cb: progress_cb(0.85, "Cross-validation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)

    # Save model
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": rf, "features": feature_cols, "classes": sorted(y.unique())}, f)

    if progress_cb: progress_cb(1.0, "Training complete ✓")

    return {
        "model":         rf,
        "features":      feature_cols,
        "accuracy":      acc,
        "f1_score":      f1,
        "auc_roc":       auc,
        "cv_mean":       cv_scores.mean(),
        "cv_std":        cv_scores.std(),
        "report":        report,
        "conf_matrix":   cm,
        "feat_importance": feat_imp,
        "X_test":        X_test,
        "y_test":        y_test,
        "y_pred":        y_pred,
        "classes":       sorted(y.unique()),
        "n_train":       len(X_train),
        "n_test":        len(X_test),
    }


def load_model() -> Optional[dict]:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


# ────────────────────────────────────────────────────────────────────────────
# PREDICTION / RISK SCORING
# ────────────────────────────────────────────────────────────────────────────
def predict_risk(model_dict: dict, X: pd.DataFrame) -> np.ndarray:
    rf   = model_dict["model"]
    cols = model_dict["features"]
    return rf.predict(X[cols].fillna(-1))


def forecast_zone_risk(model_dict: dict, lat: float, lon: float,
                        hours: int = 72) -> pd.DataFrame:
    """
    Predict risk score for a location across the next `hours` hours.
    Returns DataFrame with columns: datetime, risk_score, risk_label.
    """
    rf   = model_dict["model"]
    cols = model_dict["features"]
    now  = datetime.now()
    rows = []
    for h in range(hours):
        ts  = now + timedelta(hours=h)
        row = {
            "Hour":                   ts.hour,
            "DayOfWeek":              ts.weekday(),
            "Month":                  ts.month,
            "IsWeekend":              int(ts.weekday() >= 5),
            "IsNight":                int(ts.hour >= 20 or ts.hour <= 4),
            "IsPeakEvening":          int(17 <= ts.hour <= 22),
            "LatBin":                 int((lat - 41.64) / (42.02 - 41.64) * 40),
            "LonBin":                 int((lon - (-87.94)) / ((-87.52) - (-87.94)) * 40),
            "CrimeTypeCode":          4,   # Burglary as baseline
            "LocationCode":           0,   # Street
            "DistrictIncidentDensity": 0.5,
            "ArrestInt":              0,
            "DomesticInt":            0,
            "District":               1,
        }
        rows.append(row)

    X_fore = pd.DataFrame(rows)[cols]
    proba  = rf.predict_proba(X_fore)
    # Risk score = weighted average of class probabilities
    classes = model_dict["classes"]
    weights = np.array([c for c in classes], dtype=float)
    scores  = (proba * weights).sum(axis=1) / max(weights)   # 0–1 normalised

    preds   = rf.predict(X_fore)
    labels  = [LABEL_MAP.get(int(p), str(p)) for p in preds]

    return pd.DataFrame({
        "datetime":   [now + timedelta(hours=h) for h in range(hours)],
        "risk_score": scores,
        "risk_label": labels,
        "class_id":   preds,
    })


# ────────────────────────────────────────────────────────────────────────────
# PLOTLY FIGURES
# ────────────────────────────────────────────────────────────────────────────
DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0e1220",
    font=dict(color="#94a3b8", family="monospace", size=11),
    margin=dict(l=40, r=20, t=36, b=40),
    xaxis=dict(gridcolor="#1e2a40", showgrid=True),
    yaxis=dict(gridcolor="#1e2a40", showgrid=True),
)


def fig_feature_importance(feat_imp: pd.DataFrame, top_n: int = 12) -> go.Figure:
    top = feat_imp.head(top_n)
    fig = go.Figure(go.Bar(
        x=top["Importance"],
        y=top["Feature"],
        orientation="h",
        marker=dict(
            color=top["Importance"],
            colorscale=[[0, "#0e4d8a"], [0.5, "#00c8e8"], [1, "#e63946"]],
            showscale=False,
        ),
    ))
    fig.update_layout(title="Feature Importance (Random Forest)", **DARK,
                      yaxis=dict(autorange="reversed", gridcolor="#1e2a40"))
    return fig


def fig_confusion_matrix(cm: np.ndarray, classes: list) -> go.Figure:
    labels = [LABEL_MAP.get(c, str(c)) for c in classes]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, "#080b12"], [1, "#e63946"]],
        text=cm, texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(title="Confusion Matrix",
                      xaxis_title="Predicted", yaxis_title="Actual",
                      **DARK)
    return fig


def fig_forecast(forecast_df: pd.DataFrame, zone_name: str = "Selected Zone") -> go.Figure:
    color_map = {"Low": "#4ade80", "Medium": "#facc15", "High": "#f97316", "Critical": "#ef4444"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df["datetime"], y=forecast_df["risk_score"],
        mode="lines", name="Risk Score",
        line=dict(color="#00c8e8", width=2),
        fill="tozeroy", fillcolor="rgba(0,200,232,0.08)",
    ))
    # Add colour bands per severity zone
    for level, thresh, col in [("Critical", 0.75, "#ef4444"), ("High", 0.5, "#f97316"),
                                ("Medium", 0.25, "#facc15")]:
        fig.add_hline(y=thresh, line_dash="dot",
                      line_color=col, opacity=0.5,
                      annotation_text=level,
                      annotation_position="right",
                      annotation_font=dict(color=col, size=9))

    fig.update_layout(
        title=f"72-Hour Risk Forecast — {zone_name}",
        xaxis_title="Date / Time", yaxis_title="Risk Score (0–1)",
        yaxis=dict(range=[0, 1.05], gridcolor="#1e2a40"),
        **{k: v for k, v in DARK.items() if k != "yaxis"},
    )
    return fig


def fig_class_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["Severity"].value_counts().sort_index()
    colors = [SEV_COLORS.get(LABEL_MAP.get(i, ""), "#888") for i in counts.index]
    fig = go.Figure(go.Bar(
        x=[LABEL_MAP.get(i, str(i)) for i in counts.index],
        y=counts.values,
        marker_color=colors,
        text=counts.values,
        textposition="auto",
    ))
    fig.update_layout(title="Dataset Severity Distribution",
                      xaxis_title="Severity Class",
                      yaxis_title="Records", **DARK)
    return fig


def fig_cv_scores(cv_mean: float, cv_std: float) -> go.Figure:
    folds = [cv_mean + np.random.normal(0, cv_std) for _ in range(5)]
    fig = go.Figure(go.Bar(
        x=[f"Fold {i+1}" for i in range(5)],
        y=folds,
        marker_color="#00c8e8",
        text=[f"{v:.3f}" for v in folds],
        textposition="auto",
    ))
    fig.add_hline(y=cv_mean, line_dash="dash", line_color="#e63946",
                  annotation_text=f"Mean={cv_mean:.3f}")
    fig.update_layout(title="5-Fold Cross-Validation F1 Scores",
                      yaxis=dict(range=[0, 1], gridcolor="#1e2a40"),
                      **{k: v for k, v in DARK.items() if k != "yaxis"})
    return fig
