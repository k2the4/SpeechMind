# Free conversation intelligence setup

SpeechMind supports Gemini API free-tier inference. No model weights or Ollama installation are required for this route.

## Recommended setup (Gemini)
1. Create an API key at https://aistudio.google.com/app/apikey
2. In PowerShell, set it for the current terminal session:
   ```powershell
   $env:GEMINI_API_KEY="PASTE_YOUR_KEY_HERE"
   ```
3. Start SpeechMind from the same PowerShell window:
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

Default model: `gemini-2.5-flash-lite`. Google's current pricing page lists free text input/output subject to account-specific limits. Free-tier content may be used to improve Google products. Do not send sensitive customer recordings through this route. See https://ai.google.dev/gemini-api/docs/pricing and https://ai.google.dev/gemini-api/docs/rate-limits.

## Offline alternative
Ollama is optional. Install it only for offline/local inference, then run `SETUP_LOCAL_LLM.bat` to pull `qwen2.5:0.5b`. Without a Gemini key or a working Ollama server, SpeechMind uses deterministic fallback.

## Safeguards
Topic and signal evidence is retained only when it appears verbatim in the original transcript. Amounts and reference identifiers are extracted locally. All interpretations require human review; ASR errors can affect downstream analysis.