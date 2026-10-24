"""
VisionAI Report Generator — PDF + Markdown session reports.
"""

from __future__ import annotations
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd


def generate_markdown(history: list, chat_msgs: list | None = None) -> str:
    lines = [
        "# VisionAI Detection Session Report\n",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n",
        f"**Total events:** {len(history)}  \n",
        f"**Total objects detected:** {sum(e.total for e in history)}\n",
        "\n---\n",
    ]

    # Summary table
    lines.append("## Summary\n")
    lines.append("| # | Time | Source | Objects | Classes | Avg Conf |\n")
    lines.append("|---|------|--------|---------|---------|----------|\n")
    for i, e in enumerate(history, 1):
        lines.append(
            f"| {i} | {e.timestamp[11:19]} | {e.source} "
            f"| {e.total} | {len(e.objects)} | {e.avg_confidence:.1%} |\n"
        )
    lines.append("\n")

    # Per-event details
    lines.append("## Detection Events\n")
    for i, e in enumerate(history, 1):
        lines.append(f"### Event {i} — {e.timestamp[11:19]} ({e.source})\n")
        top = sorted(e.counts.items(), key=lambda x: -x[1])[:10]
        for obj, cnt in top:
            lines.append(f"- **{obj}**: {cnt}\n")
        lines.append("\n")

    # Chat log
    if chat_msgs:
        lines.append("## Chat Log\n")
        for msg in chat_msgs:
            role = "**User**" if msg["role"] == "user" else "**Agent**"
            lines.append(f"{role}: {msg['content']}\n\n")

    return "".join(lines)


def generate_pdf(history: list, chat_msgs: list | None = None) -> bytes:
    """Generate a PDF report. Returns bytes."""
    try:
        from fpdf import FPDF

        class PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 14)
                self.set_text_color(124, 92, 252)
                self.cell(0, 10, "VisionAI Detection Report", align="C", new_x="LMARGIN", new_y="NEXT")
                self.set_text_color(0, 0, 0)
                self.set_font("Helvetica", "", 9)
                self.cell(0, 6, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                          align="C", new_x="LMARGIN", new_y="NEXT")
                self.ln(3)

            def footer(self):
                self.set_y(-12)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(150, 150, 150)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()

        # ── Summary ──────────────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(124, 92, 252)
        pdf.cell(0, 8, "Session Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Total detection events: {len(history)}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"Total objects detected: {sum(e.total for e in history)}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # ── Table header ─────────────────────────────────────────────────
        if history:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(40, 40, 60)
            pdf.set_text_color(255, 255, 255)
            col_w = [10, 22, 22, 22, 22, 22]
            headers = ["#", "Time", "Source", "Objects", "Classes", "Avg Conf"]
            for cw, hd in zip(col_w, headers):
                pdf.cell(cw, 7, hd, border=1, fill=True)
            pdf.ln()

            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 8)
            for i, e in enumerate(history, 1):
                fill = i % 2 == 0
                pdf.set_fill_color(245, 245, 255) if fill else pdf.set_fill_color(255, 255, 255)
                row = [str(i), e.timestamp[11:19], e.source,
                       str(e.total), str(len(e.objects)), f"{e.avg_confidence:.0%}"]
                for cw, val in zip(col_w, row):
                    pdf.cell(cw, 6, val, border=1, fill=fill)
                pdf.ln()

        pdf.ln(6)

        # ── Per-event details ─────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(124, 92, 252)
        pdf.cell(0, 8, "Detection Details", new_x="LMARGIN", new_y="NEXT")

        for i, e in enumerate(history, 1):
            if pdf.get_y() > 250:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 7, f"Event {i}  —  {e.timestamp[11:19]}  ({e.source})",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(0, 0, 0)
            top = sorted(e.counts.items(), key=lambda x: -x[1])[:8]
            for obj, cnt in top:
                pdf.cell(0, 5, f"    {obj}: {cnt}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        buf = io.BytesIO()
        pdf.output(buf)
        return buf.getvalue()

    except Exception as e:
        # Fallback: return markdown as bytes if PDF fails
        md = generate_markdown(history, chat_msgs)
        return md.encode("utf-8")
