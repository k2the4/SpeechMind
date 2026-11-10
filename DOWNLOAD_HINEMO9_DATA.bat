@echo off
setlocal
cd /d "%~dp0"

echo Downloading the HinEmo-9 Hindi-English text emotion dataset...
echo This dataset contains public YouTube comment text. Review its dataset card
echo and the platform terms before using or redistributing any underlying text.
python -m scripts.download_hinemo9
if errorlevel 1 (
    echo Dataset download failed. Review the error above.
    pause
    exit /b 1
)
echo.
echo Dataset prepared in data\hinemo9\data
pause
