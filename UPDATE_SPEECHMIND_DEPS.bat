@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Updating SpeechMind dependencies...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto failed
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo.
echo SpeechMind dependencies updated successfully.
pause
exit /b 0

:failed
echo.
echo Dependency update failed. Review the error above.
pause
exit /b 1
