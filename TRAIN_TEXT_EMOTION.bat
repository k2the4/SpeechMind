@echo off
setlocal
cd /d "%~dp0"

echo ==========================================================
echo SpeechMind - Hindi/Hinglish text-emotion training
echo ==========================================================
echo.
echo STOP: The current data-preparation output is multi-label JSONL
echo (BRIGHTER + HinEmo), but the existing scripts\train_text_emotion.py
echo trainer accepts a single-label CSV. They are not compatible.
echo.
echo The previous launcher referenced a non-existent script:
echo scripts\train_text_emotion_multilabel.py
echo and passed unsupported command-line arguments.
echo.
echo This launcher is intentionally disabled until the multi-label
echo trainer is implemented and tested against the prepared JSONL schema.
echo No training or checkpoint changes have been made.
echo.
pause
exit /b 2
