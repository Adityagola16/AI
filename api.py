"""
VisionAI REST API — Flask backend for programmatic detection access.

Endpoints:
  GET  /health           → server status
  GET  /models           → list available models
  POST /detect           → detect objects in uploaded image
  POST /detect/base64    → detect from base64-encoded image
  POST /agent/analyze    → run AI agent on detection results

Security:
  • API key auth via X-API-Key header
  • Rate limiting (60 req/min)
  • CORS enabled for local development
  • No file storage — all in-memory processing

Usage:
  python api.py
  curl -X POST http://localhost:5050/detect \
       -H "X-API-Key: visionai-dev-key" \
       -F "image=@photo.jpg"
"""

import base64
import io
import os
import time
from functools import wraps
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

load_dotenv()

# ── Lazy YOLO import (only when needed) ──────────────────────────────────────
_models: dict = {}

def get_model(name: str):
    from ultralytics import YOLO
    if name not in _models:
        p = Path(name)
        _models[name] = YOLO(str(p) if p.exists() else name)
    return _models[name]

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

API_KEY      = os.getenv("VISIONAI_API_KEY", "visionai-dev-key")
RATE_LIMIT   = 60          # requests per minute
_req_times:  list[float] = []

# ── Helpers ───────────────────────────────────────────────────────────────────
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized — invalid API key"}), 401
        return f(*args, **kwargs)
    return decorated

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        now = time.time()
        _req_times[:] = [t for t in _req_times if now - t < 60]
        if len(_req_times) >= RATE_LIMIT:
            return jsonify({"error": "Rate limit exceeded (60/min)"}), 429
        _req_times.append(now)
        return f(*args, **kwargs)
    return decorated

def pil_from_request() -> Image.Image | None:
    """Extract PIL image from multipart file or base64 JSON body."""
    if "image" in request.files:
        return Image.open(request.files["image"].stream).convert("RGB")
    data = request.get_json(silent=True) or {}
    if "image_base64" in data:
        raw = base64.b64decode(data["image_base64"])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return None

def run_detection(img: Image.Image, model_name: str,
                  conf: float, iou: float, imgsz: int) -> dict:
    mdl = get_model(model_name)
    results = mdl.predict(img, conf=conf, iou=iou, imgsz=imgsz,
                          agnostic_nms=True, verbose=False)
    r = results[0]
    detections = []
    for box in (r.boxes or []):
        cid = int(box.cls[0].item())
        detections.append({
            "object":     mdl.names[cid],
            "class_id":   cid,
            "confidence": round(float(box.conf[0].item()), 4),
            "bbox":       [round(float(v), 1) for v in box.xyxy[0].tolist()],
        })
    # Annotated image as base64
    annotated = r.plot()
    buf = io.BytesIO()
    Image.fromarray(annotated[:, :, ::-1]).save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    counts = {}
    for d in detections:
        counts[d["object"]] = counts.get(d["object"], 0) + 1

    return {
        "total":         len(detections),
        "detections":    detections,
        "counts":        counts,
        "annotated_jpg": img_b64,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "service": "VisionAI API",
        "version": "1.0.0",
        "time":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models_loaded": list(_models.keys()),
    })

@app.route("/models", methods=["GET"])
@require_api_key
def list_models():
    available = []
    for name, label in [
        ("yolo26n.pt", "yolo26n (nano, local)"),
        ("yolo11n.pt", "yolo11n (nano, local)"),
        ("yolo11s.pt", "yolo11s (small)"),
        ("yolo11m.pt", "yolo11m (medium)"),
        ("yolo11l.pt", "yolo11l (large)"),
        ("yolo11x.pt", "yolo11x (xlarge)"),
    ]:
        available.append({
            "name":       name,
            "label":      label,
            "local":      Path(name).exists(),
            "size_mb":    round(Path(name).stat().st_size / 1e6, 1)
                          if Path(name).exists() else None,
        })
    return jsonify({"models": available})

@app.route("/detect", methods=["POST"])
@require_api_key
@rate_limit
def detect():
    """
    Detect objects in an uploaded image.
    Form fields:
      image       (file)   — image file
      model       (str)    — default: yolo26n.pt
      conf        (float)  — default: 0.20
      iou         (float)  — default: 0.45
      imgsz       (int)    — default: 640
    Returns JSON with detections + base64 annotated JPEG.
    """
    img = pil_from_request()
    if img is None:
        return jsonify({"error": "No image provided. Send multipart 'image' field "
                                 "or JSON 'image_base64'."}), 400

    model_name = request.form.get("model") or \
                 (request.get_json(silent=True) or {}).get("model", "yolo26n.pt")
    conf  = float(request.form.get("conf",  0.20))
    iou   = float(request.form.get("iou",   0.45))
    imgsz = int(request.form.get("imgsz", 640))

    if conf < 0.01 or conf > 0.99:
        return jsonify({"error": "conf must be between 0.01 and 0.99"}), 400
    if imgsz not in [320, 480, 640, 800, 960, 1120, 1280]:
        return jsonify({"error": "imgsz must be one of 320,480,640,800,960,1120,1280"}), 400

    t0 = time.perf_counter()
    try:
        result = run_detection(img, model_name, conf, iou, imgsz)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["model"]        = model_name
    result["image_size"]   = list(img.size)
    return jsonify(result)

@app.route("/detect/base64", methods=["POST"])
@require_api_key
@rate_limit
def detect_base64():
    """Same as /detect but accepts JSON body with image_base64."""
    return detect()

@app.route("/agent/analyze", methods=["POST"])
@require_api_key
@rate_limit
def agent_analyze():
    """
    Run VisionAgent rule-based analysis on pre-computed detections.
    Body JSON:
      detections: [{"object": "car", "confidence": 0.82}, ...]
    Returns scene description, safety alerts, suggestions.
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import pandas as pd
        from agent import VisionAgent
    except ImportError as e:
        return jsonify({"error": f"Agent not available: {e}"}), 500

    data = request.get_json(silent=True)
    if not data or "detections" not in data:
        return jsonify({"error": "Body must contain 'detections' list"}), 400

    rows = [{"Object": d.get("object","?"),
             "Confidence": float(d.get("confidence", 0.5))}
            for d in data["detections"]]
    df = pd.DataFrame(rows)

    ag = VisionAgent()
    resp, scene = ag.run(df, source=data.get("source", "api"))

    return jsonify({
        "scene_description": resp.scene_description,
        "safety_alerts":     resp.safety_alerts,
        "suggestions":       resp.suggestions,
        "anomalies":         resp.anomalies,
        "powered_by":        resp.powered_by,
        "categories":        scene.get("categories", {}),
        "total":             scene.get("total", 0),
        "avg_confidence":    scene.get("avg_conf", 0.0),
    })

# ══════════════════════════════════════════════════════════════════════════════
# Error handlers
# ══════════════════════════════════════════════════════════════════════════════
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found",
                    "available": ["/health","/models","/detect",
                                  "/detect/base64","/agent/analyze"]}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  VisionAI REST API")
    print("  http://localhost:5050")
    print(f"  API Key: {API_KEY}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5050, debug=False)
