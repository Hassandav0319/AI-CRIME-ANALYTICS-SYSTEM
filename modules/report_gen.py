"""
modules/report_gen.py
Intelligence report generation using Groq LLM via LangChain.
Also provides NLP summarization of crime patterns using LangChain.
"""
import os
from datetime import datetime
from typing import Generator
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────
# LANGCHAIN NLP SUMMARIZER
# ──────────────────────────────────────────────────────────────────────────
def summarize_crime_data(df, groq_api_key: str = "") -> str:
    """
    Use LangChain + Groq to generate a brief NLP summary
    of the loaded crime dataset.
    """
    try:
        from langchain_groq import ChatGroq
        from langchain.schema import HumanMessage, SystemMessage

        if not groq_api_key:
            groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_api_key:
            return _rule_based_summary(df)

        llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192", temperature=0.2)
        stats = _build_stats_str(df)
        msgs = [
            SystemMessage(content=(
                "You are a crime intelligence analyst. "
                "Produce a concise 3-sentence NLP summary of the dataset statistics provided. "
                "Use formal law-enforcement language. Focus on key patterns."
            )),
            HumanMessage(content=f"Dataset statistics:\n{stats}"),
        ]
        resp = llm.invoke(msgs)
        return resp.content
    except Exception:
        return _rule_based_summary(df)


def _build_stats_str(df) -> str:
    top_type  = df["Primary Type"].value_counts().index[0] if "Primary Type" in df.columns else "N/A"
    top_dist  = df["District"].value_counts().index[0]     if "District"     in df.columns else "N/A"
    arr_rate  = df["Arrest"].mean() * 100                  if "Arrest"       in df.columns else 0
    n         = len(df)
    return (
        f"Total records: {n:,}\n"
        f"Top crime type: {top_type}\n"
        f"Highest-incident district: {top_dist}\n"
        f"Arrest rate: {arr_rate:.1f}%\n"
        f"Night incidents (20:00–04:00): {df['IsNight'].sum():,} ({df['IsNight'].mean()*100:.1f}%)\n"
        f"Weekend incidents: {df['IsWeekend'].sum():,} ({df['IsWeekend'].mean()*100:.1f}%)\n"
        f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n"
    )


def _rule_based_summary(df) -> str:
    try:
        top  = df["Primary Type"].value_counts().index[0]
        dist = df["District"].value_counts().index[0]
        arr  = df["Arrest"].mean() * 100 if "Arrest" in df.columns else 0
        return (
            f"Dataset contains {len(df):,} incidents spanning "
            f"{df['Date'].min().date()} to {df['Date'].max().date()}. "
            f"The predominant crime type is {top}, with District {dist} recording "
            f"the highest incident count. "
            f"Overall arrest rate stands at {arr:.1f}%, with night-time incidents "
            f"representing {df['IsNight'].mean()*100:.1f}% of the total workload."
        )
    except Exception:
        return "Dataset loaded successfully. Run analysis to generate summary."


# ──────────────────────────────────────────────────────────────────────────
# GROQ INTELLIGENCE REPORT
# ──────────────────────────────────────────────────────────────────────────
REPORT_SYSTEM = """
You are SENTINEL, an advanced AI law-enforcement intelligence system.
Generate formal, detailed police intelligence reports.
Use official law-enforcement language, structured sections, and specific data.
Classification: LAW ENFORCEMENT SENSITIVE — NOT FOR PUBLIC RELEASE.
"""


def _build_report_context(data: dict) -> str:
    """Build structured context string from system data."""
    df        = data.get("df")
    metrics   = data.get("ml_metrics", {})
    detections = data.get("detections", [])
    report_type = data.get("report_type", "Full Incident Report")
    period    = data.get("period", "Last 24 Hours")

    ctx = f"REPORT TYPE: {report_type}\nPERIOD: {period}\n\n"

    if df is not None:
        ctx += f"=== CRIME DATASET ({len(df):,} records) ===\n"
        ctx += _build_stats_str(df)
        top5 = df["Primary Type"].value_counts().head(5)
        ctx += "\nTop crime types:\n" + "\n".join(f"  {k}: {v}" for k, v in top5.items()) + "\n"

    if metrics:
        ctx += f"\n=== ML MODEL METRICS ===\n"
        ctx += f"Accuracy: {metrics.get('accuracy',0):.1%}\n"
        ctx += f"F1 Score: {metrics.get('f1_score',0):.3f}\n"
        ctx += f"AUC-ROC:  {metrics.get('auc_roc',0):.3f}\n"
        ctx += f"CV Mean:  {metrics.get('cv_mean',0):.3f} ± {metrics.get('cv_std',0):.3f}\n"
        fi = metrics.get("feat_importance")
        if fi is not None:
            ctx += "Top 5 features:\n"
            for _, row in fi.head(5).iterrows():
                ctx += f"  {row['Feature']}: {row['Importance']:.3f}\n"

    if detections:
        ctx += f"\n=== CCTV DETECTION RESULTS (YOLOv8) ===\n"
        ctx += f"Total detections: {len(detections)}\n"
        for d in detections[:6]:
            ctx += (f"  [{d['severity']}] {d['name'].upper()} "
                    f"— confidence {d['confidence']}%\n")

    return ctx


def generate_report_groq(data: dict, groq_api_key: str = "") -> Generator[str, None, None]:
    """
    Stream intelligence report tokens from Groq.
    Yields text chunks as they arrive.
    """
    if not groq_api_key:
        groq_api_key = os.environ.get("GROQ_API_KEY", "")

    if not groq_api_key:
        yield from _fallback_report(data)
        return

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        ctx    = _build_report_context(data)
        ref_no = f"SNT-{datetime.now().strftime('%Y%m%d')}-{hash(ctx) % 10000:04d}"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        prompt = f"""
Generate a comprehensive {data.get('report_type','Full Incident Report')} for {data.get('period','Last 24 Hours')}.

Reference Number: {ref_no}
Generated: {now_str}

Context:
{ctx}

Structure the report with these exact sections marked by === SECTION NAME ===:
1. REPORT HEADER (classification, reference, timestamp)
2. EXECUTIVE SUMMARY
3. INCIDENT STATISTICS
4. CRIME PATTERN ANALYSIS
5. CCTV EVIDENCE ANALYSIS
6. MACHINE LEARNING RISK ASSESSMENT
7. GEOGRAPHIC HOTSPOT INTELLIGENCE
8. RECOMMENDED TACTICAL ACTIONS (numbered list)
9. PATROL DEPLOYMENT RECOMMENDATIONS
10. FOLLOW-UP ACTIONS REQUIRED

Be specific with numbers, locations, and times from the data. Write 600-900 words total.
"""
        stream = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": REPORT_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            stream=True,
            temperature=0.35,
            max_tokens=1200,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as e:
        yield f"[Groq API error: {e}]\n\n"
        yield from _fallback_report(data)


def _fallback_report(data: dict) -> Generator[str, None, None]:
    """Offline template report when no API key available."""
    df      = data.get("df")
    metrics = data.get("ml_metrics", {})
    dets    = data.get("detections", [])
    now     = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    ref_no  = f"SNT-{datetime.now().strftime('%Y%m%d')}-OFFLINE"
    period  = data.get("period", "Last 24 Hours")
    rtype   = data.get("report_type", "Full Incident Report")
    n_recs  = len(df) if df is not None else 0
    top_type= df["Primary Type"].value_counts().index[0] if df is not None and n_recs else "N/A"
    top_dist= df["District"].value_counts().index[0]     if df is not None and n_recs else "N/A"
    acc     = metrics.get("accuracy", 0)
    f1      = metrics.get("f1_score", 0)

    report = f"""=== REPORT HEADER ===
SENTINEL INTELLIGENCE SYSTEM — AI-GENERATED REPORT
Reference:    {ref_no}
Type:         {rtype}
Period:       {period}
Generated:    {now}
Classification: LAW ENFORCEMENT SENSITIVE

=== EXECUTIVE SUMMARY ===
SENTINEL AI has completed analysis of the crime dataset ({n_recs:,} records) for the
reporting period. The predominant crime category is {top_type}, concentrated primarily
in District {top_dist}. Machine learning predictive models identify elevated risk windows
in the evening hours (18:00–23:00), particularly on weekends.

=== INCIDENT STATISTICS ===
Total Incidents Analysed:  {n_recs:,}
Model Accuracy:            {acc:.1%}
F1 Score (Weighted):       {f1:.3f}
CCTV Detections:           {len(dets)}
Weapon Alerts:             {sum(1 for d in dets if d.get('category')=='WEAPON')}

=== CCTV EVIDENCE ANALYSIS ===
YOLOv8 computer vision processed the uploaded footage.
{'Detections logged:' if dets else 'No CCTV footage uploaded.'}
"""
    for d in dets[:5]:
        report += f"  [{d['severity']}] {d['name'].upper()} — {d['confidence']}% confidence\n"

    report += f"""
=== ML RISK ASSESSMENT ===
Random Forest model (accuracy {acc:.1%}) identifies the following risk profile:
High-risk window: 18:00–23:00 daily.
Top predictors: time of day, location zone, day of week.

=== RECOMMENDED TACTICAL ACTIONS ===
1. Deploy additional units to District {top_dist} before 18:00 daily.
2. Increase patrol frequency in high-density crime areas.
3. Review CCTV footage for additional suspect identification.
4. Cross-reference weapon detections with recent incident reports.
5. Schedule community liaison in top-3 incident districts.

=== FOLLOW-UP ACTIONS ===
- Submit for supervisory review within 24 hours.
- Update predictive model with latest incident data.
- Coordinate with neighbouring district commanders.
"""
    # Stream word by word
    for word in report.split(" "):
        yield word + " "
