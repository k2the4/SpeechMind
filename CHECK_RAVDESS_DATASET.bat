@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo SpeechMind - RAVDESS dataset check
echo ==========================================

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0scripts\train_voice_emotion.py" --prepare-only
) else (
    py -3 "%~dp0scripts\train_voice_emotion.py" --prepare-only
)

if errorlevel 1 (
    echo.
    echo Dataset check failed. Read the error above.
) else (
    echo.
    echo Dataset inventory completed.
    echo See training_outputs\ravdess_voice\dataset_summary.json
)

echo.
pause
endlocal
