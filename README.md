---
title: SpeechMind
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: gradio_app.py
python_version: 3.11
pinned: false
---

# SpeechMind

SpeechMind is an experimental multilingual speech-understanding application for Hindi, Hindi-English (Hinglish), and English recordings.

The hosted Gradio profile uses Whisper for English/Hinglish fallback and IndicConformer for Hindi/other supported Indic languages. It loads the fine-tuned MuRIL and WavLM checkpoints from the private Hugging Face model repository using the Space secrets `SPEECHMIND_MODEL_REPO` and `HF_TOKEN`.

## Hosted profile limitations

The free Gradio Space does not run the repository's separate `.venv_srota` worker. Srota ASR and the isolated IndicTrans2 worker are therefore disabled in this hosted profile. Hinglish transcription falls back to Whisper; English meaning may be produced by Gemini when `GEMINI_API_KEY` is configured. Without Gemini, deterministic conversation analysis is used and does not claim to translate.

The local Streamlit application and its existing routing are retained for local use.

## Space configuration

- Variable: `SPEECHMIND_MODEL_REPO=Hyundai-Creta/speechmind-models`
- Secret: `HF_TOKEN` (read access to the private model repository)
- Secret: `GEMINI_API_KEY` (optional)

Do not commit credentials or model checkpoints to Git.

## Limitations and safety

The public demo should only be used with non-sensitive recordings. Emotion predictions are experimental, uncalibrated and not medical or psychological diagnoses. Verify the transcript and evidence before relying on conversation insights.
