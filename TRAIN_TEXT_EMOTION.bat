@echo off
setlocal
cd /d "%~dp0"

if not exist "data\text_emotion\text_emotion.csv" (
    echo Missing data\text_emotion\text_emotion.csv
    echo Create a labeled CSV with text,emotion,language columns.
    echo See docs\TEXT_EMOTION_TRAINING.md for format and data guidance.
    pause
    exit /b 1
)

echo Training SpeechMind MuRIL text-emotion checkpoint...
python -m scripts.train_text_emotion --csv data/text_emotion/text_emotion.csv
if errorlevel 1 (
    echo Text-emotion training failed. Review the error above.
    pause
    exit /b 1
)

echo.
echo Training completed. Restart Streamlit to load the new checkpoint.
pause
