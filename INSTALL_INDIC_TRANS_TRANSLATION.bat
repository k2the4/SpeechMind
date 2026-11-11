@echo off
setlocal
cd /d "%~dp0"

set "WORKER_PY=%~dp0.venv_srota\Scripts\python.exe"
set "APP_PY=%~dp0.venv\Scripts\python.exe"

echo ==========================================================
echo SpeechMind - Install isolated IndicTrans2 translation
echo ==========================================================
echo This installs IndicTransToolkit only into .venv_srota.
echo The main .venv and its Transformers 4.46.3 stack are untouched.
echo.
echo Before the first model download, accept access to:
echo https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M
echo If the repository requests authentication, run:
echo "%~dp0.venv_srota\Scripts\hf.exe" auth login
echo.

if not exist "%WORKER_PY%" (
    echo Srota environment not found. Run SETUP_HINGLISH_ASR_ENV.bat once.
    pause
    exit /b 1
)
if not exist "%APP_PY%" (
    echo SpeechMind main environment not found.
    pause
    exit /b 1
)

"%WORKER_PY%" -m pip install -r "%~dp0requirements_indic_translation.txt"
if errorlevel 1 goto failed

"%WORKER_PY%" -c "import transformers; assert transformers.__version__ == '4.57.6', 'Expected Transformers 4.57.6'; from IndicTransToolkit.processor import IndicProcessor; print('Transformers:', transformers.__version__); print('IndicProcessor: OK')"
if errorlevel 1 goto failed

echo.
echo Toolkit installed. Starting one English-meaning translation smoke test.
"%APP_PY%" "%~dp0scripts\test_indictrans_translation.py"
if errorlevel 1 goto failed

echo.
echo IndicTrans2 translation smoke test passed.
pause
exit /b 0

:failed
echo.
echo IndicTrans2 setup or smoke test failed.
echo The existing SpeechMind .venv was not modified.
echo Review the error above. The toolkit upstream does not currently test Windows.
pause
exit /b 1
