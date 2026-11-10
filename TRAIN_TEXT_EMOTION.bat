@echo off
setlocal
cd /d "%~dp0"

echo SpeechMind's primary Hindi-English-Hinglish text-emotion workflow is HinEmo-9.
echo.
echo For the main Hinglish model:
echo   1. DOWNLOAD_HINEMO9_DATA.bat
echo   2. TRAIN_HINEMO9_TEXT_EMOTION.bat
echo.
echo The old seven-label MuRIL trainer remains available for separate
echo additional-language experiments once a suitable labeled dataset exists.
echo It is not the default Hinglish text-emotion trainer.
pause
