# Deploy SpeechMind on Streamlit Community Cloud

This guide is for the standard Streamlit Community Cloud runtime. It is separate from the Docker-based Hugging Face Spaces deployment documented in [DEPLOY_HUGGINGFACE_SPACES.md](DEPLOY_HUGGINGFACE_SPACES.md).

## App configuration

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create or open the app.
3. Select repository `k2the4/SpeechMind`.
4. Select branch `feature/speechmind-product-architecture-v2` (or the release branch you are deploying).
5. Set the main file path to `app.py`.
6. Reboot the app after changing the deployed branch or its dependencies.

The repository includes `runtime.txt` to select Python 3.11, `packages.txt` for FFmpeg and libsndfile, and a CPU-only PyTorch pin in `requirements.txt`. These are required for the supported Streamlit Community Cloud configuration and MP3/M4A decoding.

## Optional Streamlit secrets

Open the app's settings and add secrets only when the corresponding feature is required:

```toml
GEMINI_API_KEY = "your-google-ai-studio-key"
SPEECHMIND_MODEL_REPO = "your-huggingface-username/speechmind-models"
HF_TOKEN = "your-huggingface-read-token"
```

- `GEMINI_API_KEY` enables Gemini conversation analysis and the optional audio-transcription fallback. When enabled, the recording or transcript may be sent to Google's API. Do not use this with sensitive recordings unless the data-handling terms are acceptable.
- `SPEECHMIND_MODEL_REPO` and `HF_TOKEN` allow the app to download the trained MuRIL and voice-emotion checkpoints. Keep the model repository private if the weights or associated material should not be public.
- If these secrets are absent, the app should still start. Conversation analysis uses deterministic transcript rules, and emotion branches report their actual availability rather than pretending that missing checkpoints loaded.

Never commit API keys or Hugging Face tokens to GitHub.

## Runtime scope and model limitations

Streamlit Community Cloud installs the main environment from the root `requirements.txt`; it does not create the separate `.venv_srota` worker environment used by the Docker deployment. Consequently, the isolated Srota and IndicTrans2 worker paths are not guaranteed to be available in this hosting mode.

For a lightweight hosted demonstration, select **Fast support case intake**. It uses the single-pass Whisper route and skips local translation and emotion inference for that run. If Gemini is configured, it can provide contextual analysis and may be used as an audio-transcription fallback only when Whisper returns a valid, explicitly low quality score. An unknown quality score does not trigger an external audio upload.

For the full specialist ASR, IndicTrans2 and trained emotion workflow, use the Docker deployment and provision its isolated worker environment and model checkpoints. Do not describe the standard Streamlit Community Cloud app as running those worker branches unless the runtime diagnostics confirm they are available.

## Smoke test after reboot

1. Confirm the app page renders before uploading audio.
2. Upload one short WAV recording and run Fast support case intake.
3. Confirm that a transcript is produced and the original Whisper hypothesis is retained if an audio fallback is used.
4. Test an MP3 and an M4A file to verify FFmpeg decoding.
5. Confirm that conversation analysis clearly reports Gemini or deterministic fallback.
6. Open Technical diagnostics and check the selected ASR model, quality status, fallback reason and timing values.
7. Download the JSON report and verify that it contains both the ASR analysis and conversation insights.

Use non-sensitive sample recordings for public demos. SpeechMind's emotion outputs and transcript-quality heuristics are experimental and require human verification.
