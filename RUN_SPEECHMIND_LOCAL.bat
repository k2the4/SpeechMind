@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Starting SpeechMind with LOCAL models.
echo No paid API or external ASR provider is enabled.
echo.
"%PYTHON_EXE%" -m streamlit run "%~dp0app.py"
if errorlevel 1 (
    echo.
    echo SpeechMind stopped with an error. Review the message above.
    pause
)
