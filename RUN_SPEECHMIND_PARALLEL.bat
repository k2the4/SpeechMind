@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "TEST_FAILURE=0"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Main SpeechMind environment not found. Run SETUP_SECOND_PC.bat.
    exit /b 1
)

echo ==========================================================
echo SPEECHMIND - PARALLEL WORKSTREAM RUN
echo ==========================================================
echo.

echo [1/4] Checking whether voice training can run independently...
if exist "%~dp0data\ravdess" if exist "%~dp0data\crema_d\AudioWAV" (
    echo RAVDESS and CREMA-D found. Launching voice training in a separate window.
    echo Voice training log: training_multicorpus.log
    start "SpeechMind Voice Training" cmd /k call "%~dp0TRAIN_VOICE_EMOTION.bat"
) else (
    echo Voice training not launched: required local datasets are missing.
    if not exist "%~dp0data\ravdess" echo Missing: data\ravdess
    if not exist "%~dp0data\crema_d\AudioWAV" echo Missing: data\crema_d\AudioWAV
    echo The remaining checks will continue.
)
echo.

echo [2/4] Running repository unit tests...
"%PYTHON_EXE%" -m unittest discover -s tests -v
if errorlevel 1 (
    echo UNIT TESTS: FAILED
    set "TEST_FAILURE=1"
) else (
    echo UNIT TESTS: PASSED
)
echo.

echo [3/4] Checking the trained Hindi/Hinglish text-emotion model...
"%PYTHON_EXE%" -m scripts.check_text_emotion
if errorlevel 1 (
    echo TEXT EMOTION: FAILED
    set "TEST_FAILURE=1"
) else (
    echo TEXT EMOTION: PASSED
)
echo.

echo [4/4] Checking main pipeline and translation-worker dependencies...
"%PYTHON_EXE%" -c "import pipeline; print('PIPELINE IMPORT: PASS')"
if errorlevel 1 (
    echo PIPELINE IMPORT: FAILED
    set "TEST_FAILURE=1"
)

if exist "%~dp0.venv_srota\Scripts\python.exe" (
    "%~dp0.venv_srota\Scripts\python.exe" -c "import importlib.util; print('SROTA qwen_asr:', bool(importlib.util.find_spec('qwen_asr'))); print('IndicTransToolkit:', bool(importlib.util.find_spec('IndicTransToolkit')))"
) else (
    echo SROTA ENVIRONMENT: MISSING
)

echo.
echo ==========================================================
if "%TEST_FAILURE%"=="0" (
    echo CORE CHECKS COMPLETE: PASSED
) else (
    echo CORE CHECKS COMPLETE: SOME CHECKS FAILED - read output above
)
echo Voice training, if launched, continues in its own window.
echo ==========================================================
pause
exit /b %TEST_FAILURE%
