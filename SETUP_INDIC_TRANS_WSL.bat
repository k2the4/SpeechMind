@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "WSL_PROJECT_ROOT="

echo ==========================================================
echo SPEECHMIND - INDIC TRANS2 SETUP (WSL)
echo ==========================================================
echo.
where wsl.exe >nul 2>nul
if errorlevel 1 (
    echo WSL is not installed.
    echo Open PowerShell as Administrator and run:
    echo   wsl --install -d Ubuntu
    echo Restart Windows if requested, open Ubuntu once, create its user,
    echo then rerun this launcher.
    exit /b 2
)

for /f "usebackq delims=" %%P in (`wsl.exe --exec wslpath -a "%CD%" 2^>nul`) do set "WSL_PROJECT_ROOT=%%P"
if not defined WSL_PROJECT_ROOT (
    echo No usable default WSL distribution was found.
    echo In PowerShell run: wsl --install -d Ubuntu
    echo Then open Ubuntu once, create its Linux user, and rerun this launcher.
    exit /b 3
)

echo Project path inside WSL: %WSL_PROJECT_ROOT%
echo.
echo Installing isolated Linux dependencies. This does not change .venv
echo or .venv_srota. The model weights are downloaded by the next test.
echo.

rem Stream a CRLF-normalized copy into Bash. Do not rewrite the tracked .sh
rem file on the Windows-mounted drive; that would leave a local Git modification.
wsl.exe --exec bash -lc "tr -d '\r' < '%WSL_PROJECT_ROOT%/scripts/setup_indictrans_wsl.sh' | bash"
if errorlevel 1 (
    echo.
    echo WSL IndicTrans2 dependency setup failed.
    exit /b 4
)

if not exist "%PYTHON_EXE%" (
    echo Main SpeechMind Python environment not found.
    exit /b 5
)

echo.
echo Running real Hindi-English translation test.
echo First use downloads the IndicTrans2 200M model into the WSL cache.
"%PYTHON_EXE%" -m scripts.test_indictrans_translation
if errorlevel 1 (
    echo.
    echo Translation model test FAILED. Review the error above.
    exit /b 6
)

echo.
echo ==========================================================
echo INDIC TRANS2 TRANSLATION SETUP AND TEST PASSED
echo ==========================================================
pause
exit /b 0
