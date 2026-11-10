@echo off
setlocal
cd /d "%~dp0"

if not exist ".env" (
    copy ".env.example" ".env" >nul
)
notepad ".env"
echo.
echo Save the file and close Notepad. Do not commit .env or share your API key.
echo SpeechMind will use local Whisper if BODHAN_API_KEY is blank.
pause
