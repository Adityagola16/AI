"""
VisionAgent — Agentic AI layer for VisionAI object detection.

Architecture:
  Perception  → YOLO detections (boxes, scores, classes)
  Memory      → rolling session history + per-scene stats
  Reasoning   → Gemini LLM (or rule-based fallback)
  Tools       → analyze_scene, detect_risks, suggest_settings,
                generate_report, track_anomaly
  Action      → returns structured AgentResponse consumed by Streamlit UI
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# ── Optional Gemini import (gracefully absent) ────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionEvent:
    timestamp: str
    source: str           # "image" | "video" | "camera"
    objects: list[str]
    counts: dict[str, int]
    avg_confidence: float
    total: int
    image_size: tuple[int, int] = (0, 0)

@dataclass
class AgentResponse:
    scene_description: str         = ""
    safety_alerts:    list[str]    = field(default_factory=list)
    suggestions:      list[str]    = field(default_factory=list)
    auto_settings:    dict         = field(default_factory=dict)
    anomalies:        list[str]    = field(default_factory=list)
    narrative:        str          = ""
    tool_trace:       list[str]    = field(default_factory=list)
    confidence:       float        = 1.0
    powered_by:       str          = "rule-based"


# ═════════════════════════════════════════════════════════════════════════════
# VisionAgent
# ═════════════════════════════════════════════════════════════════════════════

class VisionAgent:
    """
    Multi-tool agentic AI that reasons over YOLO detection results.

    Tool pipeline (executed every call):
      1. analyze_scene      → label objects, build category counts
      2. detect_risks       → safety / security checks
      3. suggest_settings   → recommend conf / model / imgsz changes
      4. track_anomaly      → compare against session history
      5. (optional) llm_reason → send context to Gemini for open-ended insight
    """

    # COCO-80 category buckets
    PERSONS   = {"person"}
    VEHICLES  = {"bicycle","car","motorcycle","airplane","bus","train",
                 "truck","boat","traffic light","fire hydrant","stop sign",
                 "parking meter"}
    ANIMALS   = {"bird","cat","dog","horse","sheep","cow","elephant",
                 "bear","zebra","giraffe"}
    WEAPONS   = {"knife","scissors","baseball bat","sports ball"}
    FOOD      = {"banana","apple","sandwich","orange","broccoli","carrot",
                 "hot dog","pizza","donut","cake"}
    ELECTRONICS = {"laptop","mouse","remote","keyboard","cell phone",
                   "tv","microwave","oven","toaster","refrigerator"}

    # Risk rules: (condition_fn, severity, message)
    RISK_RULES = [
        (lambda d: d.get("person",0) > 10,            "HIGH",
         "[!] Large crowd detected ({person} persons) — monitor for congestion."),
        (lambda d: d.get("person",0) > 0 and
                   d.get("car",0)+d.get("truck",0)+d.get("bus",0) > 0, "MED",
         "[traffic] Pedestrians + vehicles in same frame — potential traffic risk."),
        (lambda d: d.get("knife",0) > 0 or d.get("scissors",0) > 0,   "HIGH",
         "[!] Sharp object detected — review context carefully."),
        (lambda d: d.get("fire hydrant",0) > 0 and d.get("car",0) > 0, "LOW",
         "[info] Car near fire hydrant — possible obstruction."),
        (lambda d: d.get("cell phone",0) > 0 and d.get("car",0) > 0,  "MED",
         "[!] Phone + vehicle detected — possible distracted driving scene."),
        (lambda d: sum(d.values()) > 30,               "LOW",
         "[info] Very dense scene ({total} objects) — consider patch detection."),
    ]

    def __init__(self, gemini_api_key: str | None = None):
        self.history: list[DetectionEvent] = []
        self.chat_history: list[dict]      = []
        self._gemini_key   = gemini_api_key
        self._gemini_model = None
        self._init_gemini()

    # ── Gemini initialisation ─────────────────────────────────────────────
    def _init_gemini(self):
        if not GEMINI_AVAILABLE or not self._gemini_key:
            return
        try:
            genai.configure(api_key=self._gemini_key)
            self._gemini_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=(
                    "You are VisionAI Agent, an expert computer vision analyst. "
                    "You receive YOLO object detection results and provide:\n"
                    "1. A vivid, accurate scene description\n"
                    "2. Safety and security observations\n"
                    "3. Actionable recommendations\n"
                    "4. Interesting insights about the scene\n"
                    "Be concise, structured, and use bullet points. "
                    "Never hallucinate objects not in the detection list."
                ),
            )
        except Exception:
            self._gemini_model = None

    def update_key(self, key: str):
        self._gemini_key = key
        self._init_gemini()

    @property
    def has_llm(self) -> bool:
        return self._gemini_model is not None

    # ═══════════════════════════════════════════════════════════════════════
    # TOOL 1 — analyze_scene
    # ═══════════════════════════════════════════════════════════════════════
    def analyze_scene(self, df: pd.DataFrame, source: str = "image") -> dict:
        """Categorise detections and build scene summary dict."""
        if df.empty:
            return {"categories": {}, "counts": {}, "dominant": "empty",
                    "total": 0, "avg_conf": 0.0}

        counts  = df["Object"].value_counts().to_dict()
        total   = len(df)
        avg_c   = df["Confidence"].mean() if "Confidence" in df.columns else 0.0

        categories: dict[str, int] = {
            "persons":    sum(counts.get(o,0) for o in self.PERSONS),
            "vehicles":   sum(counts.get(o,0) for o in self.VEHICLES),
            "animals":    sum(counts.get(o,0) for o in self.ANIMALS),
            "food":       sum(counts.get(o,0) for o in self.FOOD),
            "electronics":sum(counts.get(o,0) for o in self.ELECTRONICS),
            "other":      sum(v for k,v in counts.items()
                              if k not in (self.PERSONS | self.VEHICLES |
                                           self.ANIMALS | self.FOOD |
                                           self.ELECTRONICS)),
        }

        dominant = max(categories, key=categories.get)

        return {
            "categories": categories,
            "counts":     counts,
            "dominant":   dominant,
            "total":      total,
            "avg_conf":   round(avg_c, 3),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # TOOL 2 — detect_risks
    # ═══════════════════════════════════════════════════════════════════════
    def detect_risks(self, scene: dict) -> list[dict]:
        """Apply risk rules to scene summary."""
        alerts = []
        counts = scene.get("counts", {})
        total  = scene.get("total", 0)
        d      = {**counts, "total": total}

        for cond_fn, severity, template in self.RISK_RULES:
            try:
                if cond_fn(d):
                    msg = template.format(
                        person=counts.get("person", 0),
                        total=total,
                    )
                    alerts.append({"severity": severity, "message": msg})
            except Exception:
                pass
        return alerts

    # ═══════════════════════════════════════════════════════════════════════
    # TOOL 3 — suggest_settings
    # ═══════════════════════════════════════════════════════════════════════
    def suggest_settings(self, scene: dict,
                          current: dict | None = None) -> list[str]:
        """Recommend detection setting changes based on scene analysis."""
        suggestions = []
        total    = scene.get("total", 0)
        avg_conf = scene.get("avg_conf", 0.5)
        counts   = scene.get("counts", {})
        current  = current or {}

        # Too few detections
        if total == 0:
            suggestions.append("🔧 No detections — try lowering confidence to 0.05 and increasing imgsz to 960.")
        elif total < 3 and avg_conf < 0.4:
            suggestions.append("🔧 Low confidence detections — try `yolo11m` or larger model.")

        # Small objects present
        if counts.get("bird",0) > 0 or counts.get("cell phone",0) > 0:
            suggestions.append("🔍 Small objects detected — enable SAHI patch detection for better coverage.")

        # Dense crowd
        if total > 20:
            suggestions.append("📐 Dense scene — lower IoU to 0.30 to separate overlapping boxes.")

        # High confidence → can be more aggressive
        if avg_conf > 0.80 and total < 10:
            suggestions.append("⬇️ Avg confidence is high — you can safely lower confidence to 0.10 to find more objects.")

        # Vehicle scene
        vehicles = scene.get("categories",{}).get("vehicles",0)
        if vehicles > 3:
            suggestions.append("🚗 Vehicle-heavy scene — enable ByteTrack tracker for multi-frame tracking.")

        # Night / low confidence  
        if avg_conf < 0.35 and total > 0:
            suggestions.append("🌙 Low confidence — image may be dark or blurry. Try imgsz 1280 and TTA ON.")

        if not suggestions:
            suggestions.append("✅ Settings look optimal for this scene.")

        return suggestions

    # ═══════════════════════════════════════════════════════════════════════
    # TOOL 4 — track_anomaly
    # ═══════════════════════════════════════════════════════════════════════
    def track_anomaly(self, scene: dict, source: str) -> list[str]:
        """Compare current detection against session history for anomalies."""
        anomalies = []
        if len(self.history) < 2:
            return anomalies

        recent = self.history[-5:]
        avg_total = np.mean([e.total for e in recent])
        curr_total = scene.get("total", 0)

        if avg_total > 0:
            delta = (curr_total - avg_total) / avg_total
            if delta > 0.5:
                anomalies.append(
                    f"📈 Object count jumped +{delta:.0%} vs recent average "
                    f"({curr_total} vs avg {avg_total:.0f}) — unusual activity."
                )
            elif delta < -0.5:
                anomalies.append(
                    f"📉 Object count dropped {abs(delta):.0%} vs recent average "
                    f"({curr_total} vs avg {avg_total:.0f})."
                )

        # New class appeared
        if self.history:
            prev_classes = set()
            for e in recent:
                prev_classes.update(e.objects)
            new_classes = set(scene.get("counts",{}).keys()) - prev_classes
            if new_classes:
                anomalies.append(
                    f"🆕 New object type(s) appeared: **{', '.join(new_classes)}**"
                )

        return anomalies

    # ═══════════════════════════════════════════════════════════════════════
    # TOOL 5 — rule_narrative
    # ═══════════════════════════════════════════════════════════════════════
    def rule_narrative(self, scene: dict) -> str:
        """Generate a natural-language scene description without LLM."""
        counts = scene.get("counts", {})
        cats   = scene.get("categories", {})
        total  = scene.get("total", 0)

        if total == 0:
            return "The scene appears empty — no objects were detected above the confidence threshold."

        parts = []

        persons = cats.get("persons", 0)
        vehicles = cats.get("vehicles", 0)
        animals  = cats.get("animals", 0)

        if persons > 0:
            parts.append(f"**{persons} person{'s' if persons > 1 else ''}**")
        if vehicles > 0:
            veh_names = [k for k in counts if k in self.VEHICLES and counts[k] > 0]
            parts.append(f"**{vehicles} vehicle{'s' if vehicles > 1 else ''}** "
                         f"({', '.join(veh_names[:3])})")
        if animals > 0:
            parts.append(f"**{animals} animal{'s' if animals > 1 else ''}**")

        other = total - persons - vehicles - animals
        if other > 0:
            obj_names = [k for k in counts
                         if k not in (self.PERSONS | self.VEHICLES | self.ANIMALS)
                         and counts[k] > 0]
            parts.append(f"**{other} other object{'s' if other > 1 else ''}** "
                         f"({', '.join(obj_names[:4])})")

        narrative = "The scene contains " + ", ".join(parts) + "."

        dominant = scene.get("dominant", "")
        if dominant == "persons" and persons > 5:
            narrative += " This looks like a crowded area."
        elif dominant == "vehicles":
            narrative += " This appears to be a road or parking scene."
        elif dominant == "animals":
            narrative += " This could be outdoors — nature or farm environment."
        elif dominant == "electronics":
            narrative += " This looks like an indoor workspace or office."

        return narrative

    # ═══════════════════════════════════════════════════════════════════════
    # TOOL 6 — llm_reason (Gemini)
    # ═══════════════════════════════════════════════════════════════════════
    def llm_reason(self, scene: dict, user_question: str | None = None) -> str:
        """Send scene context to Gemini for advanced reasoning."""
        if not self.has_llm:
            return ""

        counts_str = "\n".join(f"  - {k}: {v}" for k, v in
                                sorted(scene["counts"].items(),
                                       key=lambda x: -x[1])[:20])
        prompt = f"""
Detection results:
- Total objects: {scene['total']}
- Average confidence: {scene['avg_conf']:.1%}
- Categories: {json.dumps(scene['categories'])}
- Object breakdown:
{counts_str}

{'User question: ' + user_question if user_question else 'Provide scene analysis, safety notes, and actionable insights.'}
"""
        try:
            resp = self._gemini_model.generate_content(prompt)
            return resp.text
        except Exception as e:
            return f"(Gemini error: {e})"

    # ═══════════════════════════════════════════════════════════════════════
    # TOOL 7 — chat (multi-turn conversation)
    # ═══════════════════════════════════════════════════════════════════════
    def chat(self, user_msg: str, last_scene: dict | None = None) -> str:
        """Multi-turn chat with context from latest detection."""
        if self.has_llm:
            ctx = ""
            if last_scene and last_scene.get("total", 0) > 0:
                ctx = (f"[Current scene: {last_scene['total']} objects detected — "
                       f"{json.dumps(last_scene['categories'])}]\n")
            full_msg = ctx + user_msg
            try:
                if not hasattr(self, "_chat_session") or self._chat_session is None:
                    self._chat_session = self._gemini_model.start_chat(history=[])
                resp = self._chat_session.send_message(full_msg)
                return resp.text
            except Exception as e:
                return f"Gemini error: {e}"
        else:
            # Rule-based chat fallback
            q = user_msg.lower()
            if last_scene and last_scene.get("total",0) > 0:
                counts = last_scene.get("counts", {})
                if any(w in q for w in ["how many", "count", "number"]):
                    return (f"I detected **{last_scene['total']} total objects** — "
                            + ", ".join(f"{v} {k}" for k,v in
                                        list(counts.items())[:5]) + ".")
                if any(w in q for w in ["safe", "danger", "risk"]):
                    risks = self.detect_risks(last_scene)
                    if risks:
                        return "\n".join(r["message"] for r in risks)
                    return "✅ No significant safety risks detected in this scene."
                if any(w in q for w in ["suggest", "improve", "better", "setting"]):
                    tips = self.suggest_settings(last_scene)
                    return "\n".join(tips)
                if any(w in q for w in ["describe", "what", "scene", "show"]):
                    return self.rule_narrative(last_scene)
            return ("I'm the VisionAI rule-based agent. Add a Gemini API key "
                    "for full conversational AI. I can answer questions about "
                    "detection counts, safety, and settings.")

    # ═══════════════════════════════════════════════════════════════════════
    # Main entry point
    # ═══════════════════════════════════════════════════════════════════════
    def run(self, df: pd.DataFrame, source: str = "image",
            current_settings: dict | None = None,
            user_question: str | None = None) -> AgentResponse:
        """
        Full agentic pipeline:
        Tool 1 → analyze_scene
        Tool 2 → detect_risks
        Tool 3 → suggest_settings
        Tool 4 → track_anomaly
        Tool 5 → rule_narrative  (always)
        Tool 6 → llm_reason      (if Gemini available)
        """
        resp      = AgentResponse()
        trace     = []

        # ── Tool 1: Scene analysis ────────────────────────────────────────
        trace.append("🔍 **Tool 1**: `analyze_scene` — categorising objects…")
        scene = self.analyze_scene(df, source)

        # ── Tool 2: Risk detection ────────────────────────────────────────
        trace.append("🛡️ **Tool 2**: `detect_risks` — checking safety rules…")
        risks = self.detect_risks(scene)
        resp.safety_alerts = [r["message"] for r in risks]

        # ── Tool 3: Settings suggestions ─────────────────────────────────
        trace.append("⚙️ **Tool 3**: `suggest_settings` — tuning recommendations…")
        resp.suggestions = self.suggest_settings(scene, current_settings)

        # ── Tool 4: Anomaly tracking ──────────────────────────────────────
        trace.append("📊 **Tool 4**: `track_anomaly` — comparing vs history…")
        resp.anomalies = self.track_anomaly(scene, source)

        # ── Tool 5: Narrative ─────────────────────────────────────────────
        trace.append("✍️ **Tool 5**: `rule_narrative` — building scene description…")
        resp.scene_description = self.rule_narrative(scene)

        # ── Tool 6: Gemini reasoning ──────────────────────────────────────
        if self.has_llm:
            trace.append("🤖 **Tool 6**: `llm_reason` — sending to Gemini 1.5 Flash…")
            resp.narrative  = self.llm_reason(scene, user_question)
            resp.powered_by = "Gemini 1.5 Flash"
        else:
            resp.narrative  = resp.scene_description
            resp.powered_by = "rule-based (add Gemini key for AI)"

        resp.tool_trace = trace

        # ── Update memory ─────────────────────────────────────────────────
        if not df.empty:
            event = DetectionEvent(
                timestamp    = datetime.now().isoformat(),
                source       = source,
                objects      = list(scene["counts"].keys()),
                counts       = scene["counts"],
                avg_confidence = scene["avg_conf"],
                total        = scene["total"],
            )
            self.history.append(event)
            if len(self.history) > 50:       # keep last 50
                self.history = self.history[-50:]

        return resp, scene
