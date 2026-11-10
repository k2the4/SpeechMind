@echo off
setlocal
cd /d "%~dp0"

if not exist "data\hinemo9\data\train.jsonl" (
    echo HinEmo-9 data not found. Run DOWNLOAD_HINEMO9_DATA.bat first.
    pause
    exit /b 1
)

echo Training the Hindi-English-Hinglish text emotion baseline...
python -m scripts.train_hinemo9_text_emotion --data-dir data/hinemo9/data --output-dir checkpoints/text_emotion_hinemo9
if errorlevel 1 (
    echo Training failed. Review the error above.
    pause
    exit /b 1
)
echo.
echo Training complete. See training_outputs\hinemo9_text_emotion\test_metrics.json
echo Restart Streamlit to load the new checkpoint.
pause
