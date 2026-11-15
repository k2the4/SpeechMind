@echo off
echo SpeechMind recommended setup uses the Gemini API free tier; Ollama is optional.
echo.
echo 1. Create a key at https://aistudio.google.com/app/apikey
echo 2. In PowerShell run: $env:GEMINI_API_KEY="YOUR_KEY"
echo 3. Start the app from that same PowerShell window:
echo    .\.venv\Scripts\python.exe -m streamlit run app.py
echo.
echo No Ollama install or model download is needed for Gemini.
pause
