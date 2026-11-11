@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv_srota\Scripts\python.exe"

echo ==========================================================
echo SpeechMind - IndicTrans2 translation toolkit setup
echo ==========================================================
echo Installs only into .venv_srota. The main .venv is untouched.
echo The toolkit upstream currently documents Linux/macOS support;
echo this check determines whether it imports successfully on this Windows PC.
echo.
if not exist "%PYTHON_EXE%" (
    echo Srota environment not found. Run SETUP_HINGLISH_ASR_ENV.bat first.
    pause
    exit /b 1
)

echo [1/2] Installing IndicTransToolkit without changing dependencies...
"%PYTHON_EXE%" -m pip install --no-deps indictranstoolkit
if errorlevel 1 goto failed

echo.
echo [2/2] Checking IndicTransToolkit import...
"%PYTHON_EXE%" -c "from IndicTransToolkit import IndicProcessor; print('IndicTransToolkit import: OK')"
if errorlevel 1 (
    "%PYTHON_EXE%" -c "from IndicTransToolkit.processor import IndicProcessor; print('IndicTransToolkit processor import: OK')"
    if errorlevel 1 goto failed
)

echo.
echo Toolkit import check passed.
echo IndicTrans2 model weights will download lazily on the first Hindi translation.
echo Restart SpeechMind and test a Hindi/Hinglish clip after this.
pause
exit /b 0

:failed
echo.
echo Toolkit installation/import failed on this Windows environment.
echo The official package documentation currently lists Linux/macOS as supported.
echo No packages were installed into the main .venv.
echo Keep the error output; we will switch the translation runtime if needed.
pause
exit /b 1
