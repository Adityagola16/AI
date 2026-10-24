@echo off
title VisionAI — REST API
echo ================================================
echo  VisionAI  ^|  Flask REST API
echo  http://localhost:5050
echo  API Key: visionai-dev-key
echo ================================================
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
python api.py
pause
