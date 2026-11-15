# SpeechMind Srota HTTP worker

Deploy this folder as a separate Hugging Face Space using the Docker SDK. It runs the same `moorlee/qwen3-asr-0.6b-hinglish` model used by the local isolated worker and keeps model weights loaded between requests. It also exposes the IndicTrans2 English-meaning endpoint used by SpeechMind's full pipeline.

## Space configuration

1. Create a new Hugging Face Space with SDK **Docker**.
2. Copy the contents of this folder into the Space repository root (`app.py`, `Dockerfile`, and `requirements.txt`).
3. In Space Settings → Variables and secrets, add `SPEECHMIND_SROTA_API_TOKEN` with a long random secret value. Optionally set `SPEECHMIND_SROTA_TORCH_THREADS` to `4`.
4. Wait for the Space to build, then verify its `/health` endpoint returns `ok: true`.
5. In Streamlit Community Cloud → Manage app → Settings → Secrets, add:

```toml
SPEECHMIND_SROTA_ENDPOINT = "https://YOUR-SPACE-NAME.hf.space"
SPEECHMIND_SROTA_API_TOKEN = "the-same-random-secret"
```

The endpoint must be the Space's `hf.space` host, without a trailing `/transcribe`. Keep the Space private if the recordings or service should not be publicly discoverable. The API token is required even for private Spaces. Never commit the token.

## Runtime notes

The worker uses CPU inference by default. The first Srota request downloads and initializes the model; later requests reuse it. IndicTrans2 is loaded only when translation is first requested. HF Space hardware determines latency and may need upgrading for production traffic. The SpeechMind client chunks recordings to fit the model's 30-second per-request limit before sending them. The `/health` endpoint reports service readiness; it does not force model downloads.
