@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "DATA_DIR=%~dp0data\text_emotion\prepared"

if not exist "%PYTHON_EXE%" (
    echo SpeechMind main environment not found. Run SETUP_SECOND_PC.bat first.
    pause
    exit /b 1
)
if not exist "%DATA_DIR%\brighter_hi_train.jsonl" (
    echo Prepared text-emotion data not found.
    echo Run PREPARE_TEXT_EMOTION_DATA.bat first.
    pause
    exit /b 1
)
if not exist "%DATA_DIR%\brighter_hi_dev.jsonl" (
    echo BRIGHTER dev split is missing. Re-run PREPARE_TEXT_EMOTION_DATA.bat.
    pause
    exit /b 1
)
if not exist "%DATA_DIR%\brighter_hi_test.jsonl" (
    echo BRIGHTER test split is missing. Re-run PREPARE_TEXT_EMOTION_DATA.bat.
    pause
    exit /b 1
)

echo ==========================================================
echo SpeechMind - Train multilingual MuRIL text-emotion model
echo ==========================================================
echo Uses BRIGHTER Hindi and HinEmo prepared JSONL.
echo Preserves official train/dev/test splits and masks unknown labels.
echo CPU partial fine-tuning: one epoch, batch 8, max length 64, final two MuRIL layers + head.
echo Uses all 2,556 BRIGHTER rows plus 1,200 sampled HinEmo rows.
echo Uses per-label positive weighting and validation-only threshold calibration.
echo Prints live progress and ETA every 25 batches.
echo.

"%PYTHON_EXE%" "%~dp0scripts\train_text_emotion_multilabel.py" --data-dir "%DATA_DIR%" --epochs 1 --batch-size 8 --max-length 64 --max-hinemo-train 1200 --max-hinemo-eval 500 --learning-rate 1e-3 --cpu-threads 4
if errorlevel 1 (
    echo.
    echo Text-emotion training failed. No new checkpoint should be used.
    pause
    exit /b 1
)

echo.
echo Training finished. Review:
echo checkpoints\text_emotion_muril\text_emotion_metrics.json
echo checkpoints\text_emotion_muril\speechmind_metadata.json
pause
exit /b 0
