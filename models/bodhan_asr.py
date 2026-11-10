"""Optional Bodhan Indic-Transcribe API adapter.

This provider is deliberately opt-in because audio is sent to a third-party
service and usage is metered. It accepts audio already normalized to mono
16 kHz float32 by SpeechMind.
"""
from __future__ import annotations

import io
import os

import numpy as np
import requests
import soundfile as sf

API_URL = "https://api.bodhan.ai/v1/audio/transcriptions"
MODEL_NAME = "indic-transcribe"
SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 30.0


class BodhanASR:
    def __init__(self, api_key: str | None = None, timeout: int = 120):
        self.api_key = api_key or os.getenv("BODHAN_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "SPEECHMIND_ASR_PROVIDER=bodhan requires BODHAN_API_KEY. "
                "Audio is sent to Bodhan AI and usage is metered."
            )
        self.timeout = timeout

    def transcribe(self, audio: np.ndarray) -> dict:
        waveform = np.asarray(audio, dtype=np.float32).squeeze()
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Bodhan ASR expects non-empty mono audio.")
        if not np.isfinite(waveform).all():
            raise ValueError("Bodhan ASR received NaN or infinite audio.")

        duration = len(waveform) / SAMPLE_RATE
        if duration > MAX_DURATION_SECONDS:
            raise ValueError(
                f"Bodhan Indic-Transcribe API accepts up to 30 seconds per "
                f"request; this recording is {duration:.1f} seconds. Use the "
                "local provider or split the recording at natural speech "
                "boundaries before using the hosted provider."
            )

        buffer = io.BytesIO()
        sf.write(buffer, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": ("speechmind.wav", buffer, "audio/wav")},
            data={"model": MODEL_NAME},
            timeout=self.timeout,
        )
        if response.status_code == 429:
            raise RuntimeError(
                "Bodhan API rate limit or credit limit reached (HTTP 429). "
                "Check the response details and Bodhan dashboard."
            )
        response.raise_for_status()
        payload = response.json()
        transcript = str(payload.get("text", "")).strip()
        if not transcript:
            raise RuntimeError("Bodhan API returned an empty transcript.")
        return {
            "transcript": transcript,
            "asr_model": "bodhan-indic-transcribe-api",
            "quality_score": None,
            "confidence": None,
            "quality_method": "not_provided_by_provider",
            "language_source": "provider_auto_code_mixed",
            "provider": "bodhan",
            "audio_duration_sec": round(duration, 3),
        }
