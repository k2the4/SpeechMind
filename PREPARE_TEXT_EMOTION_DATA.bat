@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo SpeechMind main environment not found. Run SETUP_SECOND_PC.bat first.
    pause
    exit /b 1
)

echo ==========================================================
echo SpeechMind - Prepare Hindi and Hinglish text-emotion data
echo ==========================================================
echo Downloads BRIGHTER Hindi and the CC-BY Hinglish HinEmo dataset.
echo Preserves the released train, validation and test splits.
echo Raw data is saved locally under data\text_emotion\prepared.
echo.

"%PYTHON_EXE%" -c "import datasets; print('Hugging Face datasets:', datasets.__version__)"
if errorlevel 1 (
    echo The datasets package is missing. Run UPDATE_SPEECHMIND_DEPS.bat first.
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%~dp0scripts\prepare_text_emotion_data.py"
if errorlevel 1 (
    echo Text-emotion data preparation failed. Review the error above.
    pause
    exit /b 1
)

echo.
echo Text-emotion data preparation completed.
pause
exit /b 0
