"""
modules/vision.py
YOLOv8-based weapon, person, and fight detection.
Works on uploaded images or video frames.
"""
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Union
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────
# DETECTION CONFIG
# ──────────────────────────────────────────────────────────────────────────
# YOLOv8 COCO class IDs relevant to security
RELEVANT_CLASSES = {
    0:  ("person",         "#f4a261", "MEDIUM"),
    1:  ("bicycle",        "#94a3b8", "LOW"),
    2:  ("car",            "#94a3b8", "LOW"),
    27: ("backpack",       "#94a3b8", "LOW"),
    43: ("knife",          "#e63946", "CRITICAL"),
    76: ("scissors",       "#f97316", "HIGH"),
    78: ("teddy bear",     "#94a3b8", "LOW"),
}

WEAPON_KEYWORDS  = {"knife", "scissors", "gun", "pistol", "rifle", "weapon", "blade"}
VIOLENCE_KEYWORDS = {"fight", "assault", "attack", "violence", "aggression"}
SEVERITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

# ──────────────────────────────────────────────────────────────────────────
# THEME & PALETTE (RGB)
# ──────────────────────────────────────────────────────────────────────────
PALETTE = {
    "CRITICAL": (230, 57, 70),   # #e63946 - Red
    "HIGH":     (249, 115, 22),  # #f97316 - Orange-Red
    "MEDIUM":   (244, 162, 97),  # #f4a261 - Orange
    "LOW":      (148, 163, 184), # #94a3b8 - Slate/Grey
    "PERSON":   (97, 162, 244),  # #61a2f4 - Blueish
    "VIOLENCE": (250, 139, 167), # #fa8ba7 - Pinkish
}

def get_color(category: str, severity: str) -> tuple[int, int, int]:
    """Get RGB color from palette."""
    if category == "WEAPON":   return PALETTE["CRITICAL"]
    if category == "VIOLENCE": return PALETTE["VIOLENCE"]
    if category == "PERSON":   return PALETTE["PERSON"]
    return PALETTE.get(severity, PALETTE["LOW"])



def get_model():
    """Lazy-load YOLOv8n."""
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")          # auto-downloads on first use
    except Exception as e:
        return None


def run_detection(image: Union[np.ndarray, Image.Image],
                  conf_thresh: float = 0.30,
                  model=None) -> dict:
    """
    Run YOLOv8 on a single image.
    Returns annotated PIL image + structured detection list.
    """
    if model is None:
        model = get_model()

    # Normalize input to RGB numpy array for processing
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        # If already a numpy array, assume RGB (standard for Streamlit/PIL context)
        img_rgb = image.copy()

    # Create BGR copy for OpenCV drawing
    annotated_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    detections = []

    if model is not None and model != "mock":
        # YOLOv8 expects RGB or BGR, we pass BGR to match OpenCV state
        results = model.predict(annotated_bgr, conf=conf_thresh, verbose=False)
        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                xyxy   = box.xyxy[0].cpu().numpy().astype(int)
                name   = model.names.get(cls_id, str(cls_id)).lower()

                # Determine severity & category
                if any(w in name for w in WEAPON_KEYWORDS):
                    sev, cat = "CRITICAL", "WEAPON"
                elif any(v in name for v in VIOLENCE_KEYWORDS):
                    sev, cat = "HIGH", "VIOLENCE"
                elif name == "person":
                    sev, cat = "MEDIUM", "PERSON"
                else:
                    sev, cat = "LOW", "OBJECT"

                # Get RGB color and convert to BGR for OpenCV
                rgb_color = get_color(cat, sev)
                bgr_color = rgb_color[::-1] 

                # Draw bounding box
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), bgr_color, 2)
                
                label = f"{name.upper()} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), bgr_color, -1)
                cv2.putText(annotated_bgr, label, (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                detections.append({
                    "class_id":   cls_id,
                    "name":       name,
                    "category":   cat,
                    "confidence": round(conf * 100, 1),
                    "severity":   sev,
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                    "area_px":    int((x2 - x1) * (y2 - y1)),
                })
    else:
        # Fallback – use mock (which draws on BGR)
        annotated_bgr = _mock_detections(annotated_bgr)
        detections = _mock_detection_data()

    # Sort by severity
    detections.sort(key=lambda d: SEVERITY_ORDER.get(d["severity"], 0), reverse=True)

    # Overlay metadata (BGR colours – convert RGB palette to BGR)
    meta_blue = PALETTE["PERSON"][::-1]
    meta_cyan = (232, 200, 0) # Already BGR-ish
    
    cv2.putText(annotated_bgr, "YOLOv8n  SENTINEL FEED",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, meta_cyan, 1)
    cv2.putText(annotated_bgr, f"DETS:{len(detections)}",
                (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, meta_blue, 1)

    # Final conversion TO RGB for Streamlit/PIL
    pil_out = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
    return {
        "image":       pil_out,
        "detections":  detections,
        "threat_level": _threat_level(detections),
        "n_persons":   sum(1 for d in detections if d["category"] == "PERSON"),
        "n_weapons":   sum(1 for d in detections if d["category"] == "WEAPON"),
    }



def _threat_level(detections: list) -> str:
    if any(d["severity"] == "CRITICAL" for d in detections):
        return "CRITICAL"
    if any(d["severity"] == "HIGH" for d in detections):
        return "HIGH"
    if any(d["severity"] == "MEDIUM" for d in detections):
        return "MEDIUM"
    return "LOW"


# ──────────────────────────────────────────────────────────────────────────
# MOCK (fallback when YOLO model not downloadable)
# ──────────────────────────────────────────────────────────────────────────
def _mock_detections(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    img = img.copy()
    
    # BGR colors from PALETTE
    c_person   = PALETTE["PERSON"][::-1]
    c_critical = PALETTE["CRITICAL"][::-1]
    c_violence = PALETTE["VIOLENCE"][::-1]

    boxes = [
        (int(w*.10), int(h*.20), int(w*.32), int(h*.85), "PERSON",     c_person,   "MEDIUM"),
        (int(w*.14), int(h*.35), int(w*.26), int(h*.55), "KNIFE 94%",  c_critical, "CRITICAL"),
        (int(w*.55), int(h*.25), int(w*.88), int(h*.80), "ALTERCATION",c_violence, "HIGH"),
    ]
    for x1, y1, x2, y2, label, color, _ in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(img, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img



def _mock_detection_data():
    return [
        {"class_id":43,"name":"knife","category":"WEAPON","confidence":94.1,
         "severity":"CRITICAL","bbox":[140,350,260,550],"area_px":13200},
        {"class_id":0, "name":"person","category":"PERSON","confidence":99.0,
         "severity":"MEDIUM","bbox":[100,200,320,850],"area_px":46200},
        {"class_id":0, "name":"person","category":"PERSON","confidence":96.3,
         "severity":"MEDIUM","bbox":[550,250,880,800],"area_px":82500},
    ]


# ──────────────────────────────────────────────────────────────────────────
# VIDEO FRAME EXTRACTION
# ──────────────────────────────────────────────────────────────────────────
def extract_key_frames(video_path: str, n_frames: int = 8) -> list[np.ndarray]:
    """Extract evenly spaced frames from a video file."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ──────────────────────────────────────────────────────────────────────────
# PLOTLY DETECTION SUMMARY
# ──────────────────────────────────────────────────────────────────────────
DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0e1220",
    font=dict(color="#94a3b8", family="monospace", size=11),
    margin=dict(l=40, r=20, t=36, b=40),
)


def fig_detection_summary(detections: list) -> go.Figure:
    if not detections:
        fig = go.Figure()
        fig.update_layout(title="No detections", **DARK)
        return fig

    df  = pd.DataFrame(detections)
    cat = df.groupby(["category","severity"]).size().reset_index(name="count")
    sev_colors = {"CRITICAL":"#e63946","HIGH":"#f97316","MEDIUM":"#f4a261","LOW":"#4ade80"}

    fig = go.Figure()
    for sev in ["CRITICAL","HIGH","MEDIUM","LOW"]:
        sub = cat[cat["severity"]==sev]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub["category"], y=sub["count"],
            name=sev,
            marker_color=sev_colors[sev],
            text=sub["count"], textposition="auto",
        ))
    fig.update_layout(
        title="Detection Categories & Severity",
        barmode="stack",
        xaxis=dict(title="Category", gridcolor="#1e2a40"),
        yaxis=dict(title="Count",    gridcolor="#1e2a40"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **DARK,
    )
    return fig


def fig_confidence_dist(detections: list) -> go.Figure:
    if not detections:
        return go.Figure()
    confs = [d["confidence"] for d in detections]
    names = [d["name"].upper() for d in detections]
    colors = ["#e63946" if d["severity"]=="CRITICAL"
              else "#f97316" if d["severity"]=="HIGH"
              else "#f4a261" if d["severity"]=="MEDIUM"
              else "#4ade80" for d in detections]
    fig = go.Figure(go.Bar(
        x=names, y=confs,
        marker_color=colors,
        text=[f"{c:.1f}%" for c in confs],
        textposition="auto",
    ))
    fig.update_layout(
        title="Detection Confidence Scores",
        xaxis=dict(title="Object", gridcolor="#1e2a40"),
        yaxis=dict(title="Confidence %", range=[0,105], gridcolor="#1e2a40"),
        **DARK,
    )
    return fig
