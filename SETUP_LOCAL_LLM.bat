@echo off
setlocal
cd /d "%~dp0"
echo ==========================================
echo SpeechMind - Free Local AI Setup
echo ==========================================
where ollama >nul 2>nul
if errorlevel 1 (
  echo Ollama is not installed.
  echo Install the free Windows application from:
  echo https://ollama.com/download/windows
  echo Then close and reopen PowerShell and run this file again.
  pause
  exit /b 1
)
echo.
echo Downloading Qwen 2.5 3B Instruct (quantized local model)...
ollama pull qwen2.5:3b
if errorlevel 1 (
  echo Model download failed. Check your internet connection and try again.
  pause
  exit /b 1
)
echo.
echo Checking installed model...
ollama list
echo.
echo Setup complete. Ollama should be running on http://127.0.0.1:11434
echo SpeechMind will use this model locally. No API key or paid service is needed.
pause
