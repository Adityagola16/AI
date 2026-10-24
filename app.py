"""
VisionAI — Agentic AI Object Detection Dashboard
Tabs: Image | Video | Live Camera | 🤖 AI Agent | Batch | Compare
"""

import io
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision
from PIL import Image
from ultralytics import YOLO

# ── Import local modules ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from agent import VisionAgent
from report import generate_markdown, generate_pdf

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VisionAI – Agentic Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stTabs [data-baseweb="tab"]   { font-size:1rem; padding:.5rem 1.2rem; }
.agent-box   { background:#0e1117; border:1px solid #7C5CFC;
               border-radius:10px; padding:14px; margin:6px 0; }
.tool-step   { background:#1a1a2e; border-left:3px solid #7C5CFC;
               padding:6px 12px; border-radius:4px; font-size:.83rem; margin:3px 0; }
.alert-high  { border-left:4px solid #e74c3c; background:#2d0a0a;
               padding:8px 12px; border-radius:4px; margin:4px 0; }
.alert-med   { border-left:4px solid #f39c12; background:#2d1a00;
               padding:8px 12px; border-radius:4px; margin:4px 0; }
.alert-low   { border-left:4px solid #27ae60; background:#0a2d0a;
               padding:8px 12px; border-radius:4px; margin:4px 0; }
.badge       { display:inline-block; background:#7C5CFC; color:#fff;
               border-radius:6px; padding:2px 9px; font-size:.82rem; margin:2px; }
.badge-p     { background:#e74c3c; }
.badge-v     { background:#2980b9; }
.badge-a     { background:#27ae60; }
.mode-max    { border:2px solid #f39c12; border-radius:8px; padding:8px 14px;
               background:#1a1200; color:#f39c12; font-weight:600; margin-bottom:8px; }
.powered-by  { font-size:.78rem; color:#888; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Session state init
# ══════════════════════════════════════════════════════════════════════════════
if "agent"        not in st.session_state:
    st.session_state.agent        = VisionAgent()
if "last_df"      not in st.session_state:
    st.session_state.last_df      = pd.DataFrame()
if "last_scene"   not in st.session_state:
    st.session_state.last_scene   = {}
if "chat_msgs"    not in st.session_state:
    st.session_state.chat_msgs    = []
if "last_img_rgb" not in st.session_state:
    st.session_state.last_img_rgb = None

agent: VisionAgent = st.session_state.agent

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Detection Settings")

    # ── Mode ──────────────────────────────────────────────────────────────
    st.subheader("🏆 Detection Mode")
    mode = st.radio(
        "Mode",
        ["⚡ Fast", "⚖️ Balanced", "🏆 Maximum (multi-scale + patches)"],
        index=1, label_visibility="collapsed",
    )
    max_mode = "Maximum" in mode
    balanced = "Balanced" in mode

    st.divider()

    # ── Model ─────────────────────────────────────────────────────────────
    st.subheader("🧠 Model")
    MODEL_MAP = {
        "yolo26n (nano·local)" : "yolo26n.pt",
        "yolo11n (nano·local)" : "yolo11n.pt",
        "yolo11s (small)"      : "yolo11s.pt",
        "yolo11m (medium ⭐)"  : "yolo11m.pt",
        "yolo11l (large ⭐⭐)" : "yolo11l.pt",
        "yolo11x (xlarge 🏆)"  : "yolo11x.pt",
    }
    model_label = st.selectbox("Model", list(MODEL_MAP.keys()), index=1)
    model_file  = MODEL_MAP[model_label]

    use_ensemble = st.toggle("🔀 Ensemble (2 models)", value=False)
    if use_ensemble:
        m2_opts = {k: v for k,v in MODEL_MAP.items() if v != model_file}
        model2_label = st.selectbox("2nd model", list(m2_opts.keys()), index=0)
        model2_file  = m2_opts[model2_label]
    else:
        model2_file = None

    st.divider()

    # ── Thresholds ─────────────────────────────────────────────────────────
    st.subheader("🎯 Thresholds")
    confidence = st.slider("Confidence", 0.01, 0.95, 0.05 if max_mode else 0.20, 0.01)
    iou_thresh = st.slider("IoU (NMS)",  0.10, 0.95, 0.40, 0.05)

    st.divider()

    # ── Resolution ────────────────────────────────────────────────────────
    st.subheader("🔍 Resolution")
    if max_mode:
        scales = [640, 960, 1280]
        st.info("Multi-scale: 640 + 960 + 1280")
    elif balanced:
        scales = [640, 960]
        st.info("Dual-scale: 640 + 960")
    else:
        s = st.select_slider("imgsz", [320,480,640,800,960,1280], 640)
        scales = [s]

    use_tta     = st.toggle("TTA", value=max_mode)
    use_patches = st.toggle("SAHI patches", value=max_mode)
    if use_patches:
        pg = st.selectbox("Grid", ["2×2","3×3"])
        grid_size     = (2,2) if pg == "2×2" else (3,3)
        patch_overlap = st.slider("Overlap", 0.10, 0.40, 0.20, 0.05)

    st.divider()

    # ── AI Agent ──────────────────────────────────────────────────────────
    st.subheader("🤖 AI Agent")
    gemini_key = st.text_input(
        "Gemini API key (optional)",
        type="password",
        placeholder="AIza…  (free at aistudio.google.com)",
        help="Get a free key at https://aistudio.google.com/apikey",
    )
    if gemini_key and gemini_key != getattr(agent, "_gemini_key", ""):
        agent.update_key(gemini_key)
        st.session_state.chat_msgs = []       # reset chat on new key
        st.success("✅ Gemini key set!" if agent.has_llm else "❌ Key invalid.")

    agent_auto = st.toggle("Auto-analyse after detection", value=True)

    st.divider()

    # ── Video ─────────────────────────────────────────────────────────────
    st.subheader("🎬 Video")
    tracker_choice = st.selectbox("Tracker", ["None","bytetrack.yaml","botsort.yaml"])
    max_frames     = st.number_input("Max frames", 10, 5000, 300, 10)

    st.markdown("---")
    llm_badge = "🤖 Gemini" if agent.has_llm else "🔧 Rule-based"
    st.caption(f"VisionAI • {llm_badge} Agent")

# ══════════════════════════════════════════════════════════════════════════════
# Model loaders
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_model(name: str) -> YOLO:
    p = Path(name)
    return YOLO(str(p) if p.exists() else name)

@st.cache_resource(show_spinner="Loading 2nd model…")
def load_model2(name: str) -> YOLO:
    p = Path(name)
    return YOLO(str(p) if p.exists() else name)

model  = load_model(model_file)
model2 = load_model2(model2_file) if model2_file else None

# ══════════════════════════════════════════════════════════════════════════════
# Detection helpers
# ══════════════════════════════════════════════════════════════════════════════
_CLASS_COLORS = np.random.default_rng(42).integers(80, 230, (80, 3)).tolist()

def _nms_merge(boxes, scores, classes):
    if not boxes:
        return [], [], []
    bt = torch.tensor(boxes,  dtype=torch.float32)
    st = torch.tensor(scores, dtype=torch.float32)
    keep = torchvision.ops.nms(bt, st, iou_threshold=iou_thresh).tolist()
    return [boxes[i] for i in keep], [scores[i] for i in keep], [classes[i] for i in keep]

def _predict_one(mdl, src, scale, tta):
    res = mdl.predict(src, conf=confidence, iou=iou_thresh, imgsz=scale,
                      augment=tta, agnostic_nms=True, verbose=False)
    b, s, c = [], [], []
    for box in (res[0].boxes or []):
        b.append(box.xyxy[0].tolist())
        s.append(float(box.conf[0].item()))
        c.append(int(box.cls[0].item()))
    return b, s, c

def detect_multiscale(img_pil):
    ab, as_, ac = [], [], []
    for mdl in ([model] + ([model2] if model2 else [])):
        for sc in scales:
            b, s, c = _predict_one(mdl, img_pil, sc, use_tta)
            ab += b; as_ += s; ac += c
    return _nms_merge(ab, as_, ac)

def detect_patches(img_pil):
    W, H = img_pil.size
    rows, cols = grid_size
    sx, sy = W/cols, H/rows
    px, py = sx*patch_overlap, sy*patch_overlap
    ab, as_, ac = [], [], []
    for r in range(rows):
        for c in range(cols):
            x1,y1 = max(0,int(c*sx-px)),  max(0,int(r*sy-py))
            x2,y2 = min(W,int((c+1)*sx+px)), min(H,int((r+1)*sy+py))
            patch = img_pil.crop((x1,y1,x2,y2))
            for mdl in ([model]+([model2] if model2 else [])):
                pb,ps,pc = _predict_one(mdl, patch, scales[-1], use_tta)
                for bx,sc,cl in zip(pb,ps,pc):
                    ab.append([x1+bx[0],y1+bx[1],x1+bx[2],y1+bx[3]])
                    as_.append(sc); ac.append(cl)
    return _nms_merge(ab, as_, ac)

def full_detect(img_pil):
    b, s, c = detect_multiscale(img_pil)
    if use_patches:
        pb, ps, pc = detect_patches(img_pil)
        b+=pb; s+=ps; c+=pc
        b, s, c = _nms_merge(b, s, c)
    return b, s, c

def draw_boxes(img_pil, boxes, scores, classes):
    arr = np.array(img_pil.copy())
    for box, score, cid in zip(boxes, scores, classes):
        x1,y1,x2,y2 = [int(v) for v in box]
        col = tuple(_CLASS_COLORS[cid % 80])
        name  = model.names[cid]
        label = f"{name} {score:.0%}"
        cv2.rectangle(arr, (x1,y1),(x2,y2), col, 2)
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(arr, (x1,y1-th-6),(x1+tw+4,y1), col, -1)
        cv2.putText(arr, label, (x1+2,y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return arr

def to_df(boxes, scores, classes):
    if not boxes: return pd.DataFrame()
    return pd.DataFrame([
        {"Object": model.names[cl], "Confidence": round(sc,4),
         "X1":round(b[0],1),"Y1":round(b[1],1),"X2":round(b[2],1),"Y2":round(b[3],1)}
        for b,sc,cl in zip(boxes,scores,classes)
    ]).sort_values("Confidence", ascending=False)

def png_bytes(rgb_arr):
    buf = io.BytesIO()
    Image.fromarray(rgb_arr).save(buf, format="PNG")
    buf.seek(0)
    return buf

def _badge_cls(name):
    if name == "person": return "badge-p"
    if name in {"car","truck","bus","motorcycle","bicycle","airplane","train","boat"}: return "badge-v"
    if name in {"cat","dog","bird","horse","sheep","cow","elephant","bear","zebra","giraffe"}: return "badge-a"
    return "badge"

def show_labels(df):
    if df.empty: return
    html = " ".join(
        f'<span class="{_badge_cls(r["Object"])} badge">'
        f'{r["Object"]} {r["Confidence"]:.0%}</span>'
        for _,r in df.iterrows()
    )
    st.markdown(html, unsafe_allow_html=True)

def show_analytics(df, key_pfx="x"):
    if df.empty:
        st.warning("No objects detected.")
        return
    m1,m2,m3 = st.columns(3)
    m1.metric("Detected",      len(df))
    m2.metric("Classes",       df["Object"].nunique())
    m3.metric("Avg conf",      f"{df['Confidence'].mean():.1%}")
    veh_cols = {"car","truck","bus","motorcycle","bicycle","train","boat","airplane"}
    anim_cols= {"cat","dog","bird","horse","sheep","cow","elephant","bear","zebra","giraffe"}
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("👤 Persons",  len(df[df["Object"]=="person"]))
    c2.metric("🚗 Vehicles", len(df[df["Object"].isin(veh_cols)]))
    c3.metric("🐾 Animals",  len(df[df["Object"].isin(anim_cols)]))
    c4.metric("📦 Objects",  len(df[~df["Object"].isin({"person"}|veh_cols|anim_cols)]))
    t1,t2,t3 = st.tabs(["📋 All","📊 Counts","📈 Conf dist"])
    with t1: st.dataframe(df, use_container_width=True)
    with t2: st.bar_chart(df["Object"].value_counts())
    with t3:
        # Build confidence histogram manually — avoids pd.Interval objects
        # that Altair/Vega-Lite rejects even when cast to str.
        step   = 0.05
        edges  = np.arange(0.0, 1.0 + step, step)
        labels = [f"{lo:.0%}–{hi:.0%}" for lo, hi in zip(edges[:-1], edges[1:])]
        counts = [
            int(((df["Confidence"] >= lo) & (df["Confidence"] < hi)).sum())
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        dist_df = (
            pd.DataFrame({"Confidence Range": labels, "Count": counts})
            .query("Count > 0")
            .set_index("Confidence Range")
        )
        st.bar_chart(dist_df)
    st.download_button("⬇️ CSV Report", df.to_csv(index=False).encode(),
                       "detection_report.csv", "text/csv", key=f"csv_{key_pfx}")

def run_agent(df, source="image"):
    current_settings = {
        "confidence": confidence, "iou": iou_thresh,
        "scales": scales, "model": model_label,
        "tta": use_tta, "patches": use_patches,
    }
    resp, scene = agent.run(df, source=source, current_settings=current_settings)
    st.session_state.last_df    = df
    st.session_state.last_scene = scene
    return resp, scene

def show_agent_panel(resp, key_pfx="a"):
    """Render agent response in an expander."""
    with st.expander("🤖 **AI Agent Analysis**", expanded=True):
        pwb = resp.powered_by
        st.markdown(f'<span class="powered-by">Powered by: {pwb}</span>',
                    unsafe_allow_html=True)

        # Scene description
        if resp.scene_description:
            st.markdown("### 📸 Scene")
            st.markdown(resp.scene_description)

        # Gemini narrative (if different)
        if resp.narrative and resp.narrative != resp.scene_description:
            st.markdown("### 🤖 Gemini Analysis")
            st.markdown(resp.narrative)

        # Safety alerts
        if resp.safety_alerts:
            st.markdown("### 🛡️ Safety Alerts")
            for alert in resp.safety_alerts:
                sev = "high" if "⚠️" in alert or "🔪" in alert else \
                      "med"  if "🚦" in alert or "📱" in alert else "low"
                st.markdown(f'<div class="alert-{sev}">{alert}</div>',
                            unsafe_allow_html=True)

        # Anomalies
        if resp.anomalies:
            st.markdown("### 📈 Anomalies vs History")
            for a in resp.anomalies:
                st.info(a)

        # Suggestions
        if resp.suggestions:
            st.markdown("### ⚙️ Agent Suggestions")
            for s in resp.suggestions:
                st.markdown(f"- {s}")

        # Tool trace
        with st.expander("🔬 Agent tool trace"):
            for step in resp.tool_trace:
                st.markdown(f'<div class="tool-step">{step}</div>',
                            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_img, tab_vid, tab_cam, tab_agent, tab_batch, tab_compare = st.tabs(
    ["🖼️ Image", "🎬 Video", "📷 Camera", "🤖 AI Agent", "📦 Batch", "⚖️ Compare"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — IMAGE
# ─────────────────────────────────────────────────────────────────────────────
with tab_img:
    st.header("Image Object Detection")
    if max_mode:
        st.markdown('<div class="mode-max">🏆 MAXIMUM — multi-scale + SAHI patches + ensemble</div>',
                    unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload an image",
                                 type=["jpg","jpeg","png","webp","bmp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        with st.spinner("🔍 Detecting…"):
            t0 = time.perf_counter()
            boxes, scores, classes = full_detect(image)
            elapsed = time.perf_counter() - t0

        annotated = draw_boxes(image, boxes, scores, classes)
        st.session_state.last_img_rgb = annotated

        c1, c2 = st.columns(2)
        c1.image(image,     caption="Original",
                 use_container_width=True)
        c2.image(annotated, caption=f"Detected — {len(boxes)} objects",
                 use_container_width=True)

        st.caption(f"⏱️ {elapsed:.2f}s  •  scales: {'+'.join(str(s) for s in scales)}"
                   + ("  •  TTA" if use_tta else "")
                   + ("  •  SAHI" if use_patches else ""))

        st.download_button("⬇️ Download PNG", png_bytes(annotated),
                           "detected.png", "image/png")

        df = to_df(boxes, scores, classes)
        show_labels(df)
        show_analytics(df, "img")

        if agent_auto:
            resp, _ = run_agent(df, "image")
            show_agent_panel(resp, "img")
    else:
        st.info("⬆️ Upload an image to begin.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — VIDEO
# ─────────────────────────────────────────────────────────────────────────────
with tab_vid:
    st.header("Video Object Detection")

    vf = st.file_uploader("Upload a video",
                           type=["mp4","avi","mov","mkv","webm"], key="vid")
    if vf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as ti:
            ti.write(vf.read()); tip = ti.name

        cap = cv2.VideoCapture(tip)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        pf = min(int(max_frames), total)
        st.info(f"**{total}** frames @ **{fps:.1f}** fps — processing **{pf}**")

        if st.button("▶️ Run Detection", key="run_vid"):
            to_ = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            to_.close()
            writer = cv2.VideoWriter(to_.name, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
            cap    = cv2.VideoCapture(tip)
            prog   = st.progress(0, "Processing…")
            prev   = st.empty()
            rows: list[dict] = []
            use_tr = tracker_choice if tracker_choice != "None" else None

            for fi in range(pf):
                ret, frame = cap.read()
                if not ret: break
                fp = Image.fromarray(frame[:,:,::-1])
                if use_tr:
                    res = model.track(frame, conf=confidence, iou=iou_thresh,
                                      imgsz=scales[0], agnostic_nms=True,
                                      tracker=use_tr, persist=True, verbose=False)
                    ann = res[0].plot()
                    for box in (res[0].boxes or []):
                        cid = int(box.cls[0].item())
                        r = {"Frame":fi,"Object":model.names[cid],
                             "Confidence":round(float(box.conf[0].item()),4)}
                        if box.id is not None: r["Track ID"]=int(box.id[0].item())
                        rows.append(r)
                else:
                    b,s,c = detect_multiscale(fp)
                    ann   = draw_boxes(fp, b, s, c)[:,:,::-1]
                    for bx,sc,cl in zip(b,s,c):
                        rows.append({"Frame":fi,"Object":model.names[cl],
                                     "Confidence":round(sc,4)})
                writer.write(ann)
                if fi % 30 == 0:
                    prev.image(ann[:,:,::-1], caption=f"Frame {fi+1}",
                               use_container_width=True)
                prog.progress((fi+1)/pf, text=f"Frame {fi+1}/{pf}")

            cap.release(); writer.release()
            prog.empty(); prev.empty()
            st.success(f"✅ Done! {len(rows)} total detections.")

            with open(to_.name,"rb") as f: vbytes = f.read()
            st.download_button("⬇️ Download Video", vbytes,
                               "annotated_video.mp4", "video/mp4")

            if rows:
                dv = pd.DataFrame(rows)
                show_analytics(dv[["Object","Confidence"]].copy(), "vid")
                if agent_auto:
                    resp,_ = run_agent(dv, "video")
                    show_agent_panel(resp, "vid")
                st.download_button("⬇️ CSV", dv.to_csv(index=False).encode(),
                                   "video_report.csv","text/csv",key="vcsv")
            try: os.unlink(tip); os.unlink(to_.name)
            except: pass
    else:
        st.info("⬆️ Upload a video to begin.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — CAMERA
# ─────────────────────────────────────────────────────────────────────────────
with tab_cam:
    st.header("📷 Live Camera Detection")
    st.info("Use **http://localhost:8501** for camera permission.", icon="ℹ️")

    ca, cb = st.columns([1,3])
    with ca:
        auto = st.toggle("🔁 Auto-detect", value=False)
    with cb:
        if auto: st.success("Auto-detect ON")
        else:    st.caption("Click camera icon → Take Photo")

    snap = st.camera_input("", label_visibility="collapsed",
                            key=f"cam_{int(time.time()) if auto else 'static'}")
    if snap:
        img = Image.open(snap).convert("RGB")
        with st.spinner("🔍 Detecting…"):
            t0 = time.perf_counter()
            b,s,c = full_detect(img)
            elapsed = time.perf_counter() - t0
        ann = draw_boxes(img,b,s,c)
        c1,c2 = st.columns(2)
        c1.image(img, caption="Original", use_container_width=True)
        c2.image(ann, caption=f"{len(b)} objects  •  {elapsed:.2f}s",
                 use_container_width=True)
        df = to_df(b,s,c)
        show_labels(df)
        st.download_button("⬇️ Download PNG", png_bytes(ann),
                           "camera_detect.png","image/png")
        show_analytics(df,"cam")
        if agent_auto:
            resp,_ = run_agent(df,"camera")
            show_agent_panel(resp,"cam")
        if auto:
            time.sleep(0.3); st.rerun()
    elif auto:
        time.sleep(0.5); st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — 🤖 AI AGENT
# ─────────────────────────────────────────────────────────────────────────────
with tab_agent:
    st.header("🤖 VisionAI Agent")

    # ── Status bar ────────────────────────────────────────────────────────
    llm_status = ("🟢 **Gemini 1.5 Flash** connected" if agent.has_llm
                  else "🟡 **Rule-based mode** — add Gemini key in sidebar for full AI")
    st.markdown(llm_status)
    if not agent.has_llm:
        st.markdown(
            "👉 Get a **free** Gemini API key at "
            "[aistudio.google.com/apikey](https://aistudio.google.com/apikey) "
            "and paste it in the sidebar."
        )
    st.divider()

    # ── Agent Architecture diagram ─────────────────────────────────────────
    with st.expander("📐 Agent Architecture", expanded=False):
        st.markdown("""
```
┌─────────────────────────────────────────────────────────────┐
│                    VisionAI Agentic System                  │
├──────────────┬──────────────────────────────────────────────┤
│  PERCEPTION  │  YOLO Detection (multi-scale + SAHI patches) │
├──────────────┼──────────────────────────────────────────────┤
│   TOOLS      │  Tool 1: analyze_scene   → category counts   │
│  (agentic    │  Tool 2: detect_risks    → safety alerts     │
│   loop)      │  Tool 3: suggest_settings→ auto-tune advice  │
│              │  Tool 4: track_anomaly   → vs session history│
│              │  Tool 5: rule_narrative  → scene description │
│              │  Tool 6: llm_reason      → Gemini reasoning  │
├──────────────┼──────────────────────────────────────────────┤
│   MEMORY     │  Rolling 50-event detection history          │
│              │  Multi-turn chat context                     │
├──────────────┼──────────────────────────────────────────────┤
│   REASONING  │  Gemini 1.5 Flash (or rule-based fallback)  │
├──────────────┼──────────────────────────────────────────────┤
│    ACTION    │  Alerts · Settings advice · CSV report        │
│              │  Chat answers · Anomaly notifications        │
└──────────────┴──────────────────────────────────────────────┘
```
""")

    # ── Manual run on last detection ───────────────────────────────────────
    st.subheader("📊 Analyse Last Detection")
    if st.session_state.last_df.empty:
        st.info("Upload an image / video / snapshot first — the agent will "
                "automatically analyse results.")
    else:
        df_last = st.session_state.last_df
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.session_state.last_img_rgb is not None:
                st.image(st.session_state.last_img_rgb,
                         caption="Last detection",
                         use_container_width=True)
        with col2:
            st.metric("Objects", len(df_last))
            st.metric("Classes", df_last["Object"].nunique())
            if "Confidence" in df_last.columns:
                st.metric("Avg conf", f"{df_last['Confidence'].mean():.1%}")

        if st.button("🔄 Re-run Agent Analysis"):
            with st.spinner("Agent thinking…"):
                resp, _ = run_agent(df_last)
            show_agent_panel(resp, "manual")

        st.markdown("#### Last scene summary")
        scene = st.session_state.last_scene
        if scene:
            cats = scene.get("categories", {})
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("👤 Persons",   cats.get("persons",0))
            sc2.metric("🚗 Vehicles",  cats.get("vehicles",0))
            sc3.metric("🐾 Animals",   cats.get("animals",0))
            sc4.metric("📦 Other",     cats.get("other",0))

    st.divider()

    # ── Detection history ──────────────────────────────────────────────────
    st.subheader("📜 Detection History")
    if not agent.history:
        st.caption("No detections yet.")
    else:
        hist_rows = [
            {"Time": e.timestamp[11:19], "Source": e.source,
             "Total": e.total, "Classes": len(e.objects),
             "Avg conf": f"{e.avg_confidence:.1%}",
             "Top objects": ", ".join(list(e.counts.keys())[:4])}
            for e in reversed(agent.history[-10:])
        ]
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)

        # Trend chart
        if len(agent.history) > 1:
            trend = pd.DataFrame({
                "Detection #": range(1, len(agent.history)+1),
                "Objects":     [e.total for e in agent.history],
            }).set_index("Detection #")
            st.line_chart(trend, height=150)

        if st.button("🗑️ Clear history"):
            agent.history.clear()
            st.rerun()

    st.divider()

    # ── Chat interface ─────────────────────────────────────────────────────
    st.subheader("💬 Chat with Agent")
    st.caption(
        "Ask anything about the detected scene, safety, settings, "
        "or request a detailed report."
    )

    # Display chat history
    for msg in st.session_state.chat_msgs:
        with st.chat_message(msg["role"],
                             avatar="🤖" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])

    # Quick prompts
    st.markdown("**Quick prompts:**")
    qp_cols = st.columns(4)
    quick_prompts = [
        "Describe the scene",
        "Any safety risks?",
        "How to improve detection?",
        "What objects are present?",
    ]
    chosen_qp = None
    for i, (col, qp) in enumerate(zip(qp_cols, quick_prompts)):
        if col.button(qp, key=f"qp_{i}"):
            chosen_qp = qp

    # Chat input
    user_input = st.chat_input("Ask the agent anything…") or chosen_qp

    if user_input:
        st.session_state.chat_msgs.append({"role":"user","content":user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Agent thinking…"):
                reply = agent.chat(user_input, st.session_state.last_scene)
            st.markdown(reply)

        st.session_state.chat_msgs.append({"role":"assistant","content":reply})

    st.divider()

    # ── Export full session report ─────────────────────────────────────────
    st.subheader("📄 Export Session Report")
    if agent.history:
        r1, r2 = st.columns(2)
        with r1:
            md = generate_markdown(agent.history, st.session_state.chat_msgs)
            st.download_button(
                "⬇️ Download Markdown Report",
                md.encode("utf-8"),
                file_name="visionai_report.md",
                mime="text/markdown",
            )
        with r2:
            if st.button("📄 Generate PDF Report"):
                with st.spinner("Building PDF…"):
                    pdf_bytes = generate_pdf(agent.history, st.session_state.chat_msgs)
                st.download_button(
                    "⬇️ Download PDF Report",
                    pdf_bytes,
                    file_name="visionai_report.pdf",
                    mime="application/pdf",
                    key="pdf_dl",
                )
    else:
        st.caption("Run at least one detection to generate a session report.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.header("📦 Batch Image Processing")
    st.caption("Upload multiple images at once — detect all, download a ZIP of annotated PNGs + CSV report.")

    batch_files = st.file_uploader(
        "Upload images (multiple allowed)",
        type=["jpg","jpeg","png","webp","bmp"],
        accept_multiple_files=True,
        key="batch",
    )

    if batch_files:
        st.info(f"**{len(batch_files)}** image(s) queued for detection.")
        col_b1, col_b2 = st.columns(2)
        batch_conf  = col_b1.slider("Confidence", 0.01, 0.95, confidence, 0.01, key="b_conf")
        batch_imgsz = col_b2.select_slider("imgsz", [320,480,640,960,1280], 640, key="b_imgsz")

        if st.button("▶️ Run Batch Detection", key="run_batch"):
            all_rows     = []
            zip_buf      = io.BytesIO()
            prog         = st.progress(0, "Batch processing…")
            preview_cols = st.columns(min(len(batch_files), 4))
            n            = len(batch_files)

            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, bf in enumerate(batch_files):
                    img = Image.open(bf).convert("RGB")
                    res = model.predict(img, conf=batch_conf, imgsz=batch_imgsz,
                                        agnostic_nms=True, verbose=False)
                    r   = res[0]
                    ann = draw_boxes(img,
                                     [b.xyxy[0].tolist() for b in (r.boxes or [])],
                                     [float(b.conf[0].item()) for b in (r.boxes or [])],
                                     [int(b.cls[0].item())   for b in (r.boxes or [])])

                    # Save annotated PNG into ZIP
                    ann_buf = io.BytesIO()
                    Image.fromarray(ann).save(ann_buf, format="PNG")
                    safe_name = Path(bf.name).stem + "_detected.png"
                    zf.writestr(safe_name, ann_buf.getvalue())

                    # Collect rows
                    for box in (r.boxes or []):
                        cid = int(box.cls[0].item())
                        all_rows.append({
                            "File":       bf.name,
                            "Object":     model.names[cid],
                            "Confidence": round(float(box.conf[0].item()), 4),
                            "X1": round(float(box.xyxy[0][0].item()),1),
                            "Y1": round(float(box.xyxy[0][1].item()),1),
                            "X2": round(float(box.xyxy[0][2].item()),1),
                            "Y2": round(float(box.xyxy[0][3].item()),1),
                        })

                    # Preview in grid
                    col_idx = idx % 4
                    preview_cols[col_idx].image(ann, caption=bf.name,
                                                use_container_width=True)

                    prog.progress((idx+1)/n, text=f"Processing {idx+1}/{n}: {bf.name}")

                # Add CSV to ZIP
                if all_rows:
                    csv_buf = pd.DataFrame(all_rows).to_csv(index=False).encode()
                    zf.writestr("batch_detection_report.csv", csv_buf)

            zip_buf.seek(0)
            prog.empty()
            st.success(f"✅ Done! {len(all_rows)} total detections across {n} images.")

            st.download_button(
                "⬇️ Download ZIP (annotated images + CSV)",
                zip_buf.getvalue(),
                file_name="visionai_batch_results.zip",
                mime="application/zip",
            )

            if all_rows:
                df_batch = pd.DataFrame(all_rows)
                show_analytics(df_batch[["Object","Confidence"]], "batch")

                # Run agent on batch
                if agent_auto:
                    resp, _ = run_agent(df_batch, "batch")
                    show_agent_panel(resp, "batch")
    else:
        st.info("⬆️ Upload multiple images using the picker above (hold Ctrl/Cmd to select many).")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.header("⚖️ Model Comparison")
    st.caption("Run two different YOLO models on the same image side-by-side — compare accuracy and speed.")

    cmp_file = st.file_uploader("Upload an image to compare", type=["jpg","jpeg","png","webp"], key="cmp")

    cmp_c1, cmp_c2 = st.columns(2)
    with cmp_c1:
        st.subheader("Model A")
        cmp_mA_label = st.selectbox("Model A", list(MODEL_MAP.keys()), index=0, key="cmpA")
        cmp_mA = MODEL_MAP[cmp_mA_label]
    with cmp_c2:
        st.subheader("Model B")
        cmp_mB_label = st.selectbox("Model B", list(MODEL_MAP.keys()), index=2, key="cmpB")
        cmp_mB = MODEL_MAP[cmp_mB_label]

    cmp_conf  = st.slider("Confidence (both)", 0.01, 0.95, 0.20, 0.01, key="cmp_conf")
    cmp_imgsz = st.select_slider("imgsz (both)", [320,480,640,960,1280], 640, key="cmp_imgsz")

    if cmp_file and st.button("▶️ Compare Models", key="run_cmp"):
        img_cmp = Image.open(cmp_file).convert("RGB")

        with st.spinner(f"Running {cmp_mA_label.split('(')[0].strip()}…"):
            mA  = load_model(cmp_mA)
            t0A = time.perf_counter()
            rA  = mA.predict(img_cmp, conf=cmp_conf, imgsz=cmp_imgsz,
                             agnostic_nms=True, verbose=False)[0]
            tA  = time.perf_counter() - t0A
            bA  = [b.xyxy[0].tolist() for b in (rA.boxes or [])]
            sA  = [float(b.conf[0].item()) for b in (rA.boxes or [])]
            cA  = [int(b.cls[0].item())   for b in (rA.boxes or [])]
            annA = draw_boxes(img_cmp, bA, sA, cA)

        with st.spinner(f"Running {cmp_mB_label.split('(')[0].strip()}…"):
            mB  = load_model(cmp_mB)
            t0B = time.perf_counter()
            rB  = mB.predict(img_cmp, conf=cmp_conf, imgsz=cmp_imgsz,
                             agnostic_nms=True, verbose=False)[0]
            tB  = time.perf_counter() - t0B
            bB  = [b.xyxy[0].tolist() for b in (rB.boxes or [])]
            sB  = [float(b.conf[0].item()) for b in (rB.boxes or [])]
            cB_ = [int(b.cls[0].item())   for b in (rB.boxes or [])]
            annB = draw_boxes(img_cmp, bB, sB, cB_)

        # Side-by-side display
        rc1, rc2 = st.columns(2)
        with rc1:
            nm = cmp_mA_label.split('(')[0].strip()
            st.image(annA, caption=f"{nm} — {len(bA)} objects  •  {tA:.2f}s",
                     use_container_width=True)
            st.download_button("⬇️ Download A", png_bytes(annA),
                               f"detected_{nm}.png","image/png", key="dl_A")
        with rc2:
            nm = cmp_mB_label.split('(')[0].strip()
            st.image(annB, caption=f"{nm} — {len(bB)} objects  •  {tB:.2f}s",
                     use_container_width=True)
            st.download_button("⬇️ Download B", png_bytes(annB),
                               f"detected_{nm}.png","image/png", key="dl_B")

        # Metrics comparison
        st.subheader("📊 Comparison Metrics")
        dfA = to_df(bA, sA, cA)
        dfB = to_df(bB, sB, cB_)

        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric(f"A: Objects",  len(bA))
        mc2.metric(f"B: Objects",  len(bB),
                   delta=len(bB)-len(bA),
                   delta_color="normal")
        mc3.metric(f"A: Classes",  dfA["Object"].nunique() if not dfA.empty else 0)
        mc4.metric(f"B: Classes",  dfB["Object"].nunique() if not dfB.empty else 0,
                   delta=(dfB["Object"].nunique() if not dfB.empty else 0) -
                         (dfA["Object"].nunique() if not dfA.empty else 0),
                   delta_color="normal")
        mc5.metric("A: Time", f"{tA:.2f}s")
        mc6.metric("B: Time", f"{tB:.2f}s",
                   delta=f"{tB-tA:+.2f}s",
                   delta_color="inverse")

        # Objects detected by each
        st.subheader("Detections per Model")
        set_A = set(dfA["Object"].unique() if not dfA.empty else [])
        set_B = set(dfB["Object"].unique() if not dfB.empty else [])
        only_A = set_A - set_B
        only_B = set_B - set_A
        both   = set_A & set_B

        vc1, vc2, vc3 = st.columns(3)
        vc1.markdown("**Only in A:**\n" + ("\n".join(f"- {o}" for o in sorted(only_A)) or "None"))
        vc2.markdown("**In both:**\n"   + ("\n".join(f"- {o}" for o in sorted(both))   or "None"))
        vc3.markdown("**Only in B:**\n" + ("\n".join(f"- {o}" for o in sorted(only_B)) or "None"))

        # Full side-by-side dataframes
        with st.expander("Full detection tables"):
            tc1, tc2 = st.columns(2)
            tc1.dataframe(dfA, use_container_width=True)
            tc2.dataframe(dfB, use_container_width=True)

    elif not cmp_file:
        st.info("⬆️ Upload an image to compare two models side-by-side.")

