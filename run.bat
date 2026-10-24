@echo off
title VisionAI — YOLO Object Detection
echo ================================================
echo  VisionAI  •  YOLO Object Detection Dashboard
echo ================================================
echo.
echo Starting Streamlit on http://localhost:8501
echo Press Ctrl+C to stop.
echo.
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
streamlit run app.py --server.port 8501 --server.headless true
pause
