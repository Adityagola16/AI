@echo off
title VisionAI — Dashboard
echo ================================================
echo  VisionAI  ^|  Streamlit Dashboard
echo  http://localhost:8501
echo ================================================
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
streamlit run app.py --server.port 8501 --server.headless true
pause
