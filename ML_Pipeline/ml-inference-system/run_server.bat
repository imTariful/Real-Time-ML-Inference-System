@echo off
REM Batch script to run the ML Inference System using venv

cd /d "D:\ai_\ML_Pipeline\ml-inference-system"

REM Activate venv
call "D:\ai_\ML_Pipeline\.venv\Scripts\activate.bat"

REM Run the server
python simple_server.py

pause
