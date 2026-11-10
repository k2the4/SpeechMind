@echo off
setlocal
cd /d "%~dp0"

echo Installing the FastText runtime required by IndicLID...
python -m pip install fasttext-wheel
if errorlevel 1 (
    echo fasttext-wheel installation failed. See the setup guide.
    pause
    exit /b 1
)

echo Downloading AI4Bharat IndicLID native and romanized models...
python -m scripts.download_indiclid
if errorlevel 1 (
    echo IndicLID model download failed.
    pause
    exit /b 1
)

echo.
echo IndicLID is installed. Restart SpeechMind.
pause
