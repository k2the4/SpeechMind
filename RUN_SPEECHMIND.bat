@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run app.py
if errorlevel 1 (
    echo SpeechMind stopped with an error. Review the message above.
    pause
)
