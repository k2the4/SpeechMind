@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo SpeechMind main virtual environment not found.
    echo Run the existing project setup before continuing.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "scripts\check_hinglish_english_route.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo English/Hinglish route check failed. See the error above.
) else (
    echo English/Hinglish route check completed.
)
pause
exit /b %EXIT_CODE%
