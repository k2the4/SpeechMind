# Free local conversation analysis

SpeechMind can use a local open-source language model for contextual conversation
analysis. It does not require a cloud API, API key, subscription or per-call fee.

## Windows setup

1. Install the free Ollama Windows application: https://ollama.com/download/windows
2. Close and reopen PowerShell.
3. From the SpeechMind repository, run `SETUP_LOCAL_LLM.bat`.
4. The script downloads `qwen2.5:0.5b` (a small quantized 0.5B instruction model).
5. Start SpeechMind normally with `streamlit run app.py`.

Ollama runs at `http://127.0.0.1:11434`. SpeechMind sends the transcript to
that local endpoint only. Model weights are downloaded once; subsequent
analysis works offline while Ollama is running.

## Configuration

- Default model: `qwen2.5:0.5b`
- Override model: set `SPEECHMIND_LLM_MODEL` before launching Streamlit.
- Override endpoint: set `SPEECHMIND_OLLAMA_URL` if Ollama uses another local address.
- If Ollama is unavailable, the app falls back to deterministic rules and shows a warning.

## Safety and scope

The local model generates English meaning, contextual topics, a summary,
review cues, missing-information questions and a next step. SpeechMind validates
topic and signal evidence against verbatim transcript substrings. Amounts and
reference identifiers are extracted from the original transcript with local
regular expressions rather than accepted from free-form model output.

The model is an assistant, not a source of verified case outcomes. Human review
is required. ASR errors can still affect every downstream result. Do not treat
model-generated translation as a substitute for listening to the original audio.
