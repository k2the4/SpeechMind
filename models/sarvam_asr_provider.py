"""Optional hosted ASR provider using Sarvam Saaras v4.

The caller must explicitly choose this provider. Audio is sent to Sarvam AI;
the local Srota/Whisper/IndicConformer path remains the default.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf


class SarvamASRProvider:
    MODEL_NAME = "saaras:v4"
    MODE = "codemix"
    MAX_CLIP_SECONDS = 30.0

    def __init__(self, api_key: str | None = None, client=None):
        self.api_key = (
            api_key
            or os.getenv("SARVAM_API_KEY")
            or os.getenv("SARVAM_API_SUBSCRIPTION_KEY")
        )
        self._client = client
        if self._client is None and not self.api_key:
            raise RuntimeError(
                "Hosted Saaras ASR requires SARVAM_API_KEY. "
                "Set it locally or in the deployment secrets."
            )

    def _get_client(self):
        if self._client is None:
            try:
                from sarvamai import SarvamAI
            except ImportError as error:
                raise RuntimeError(
                    "The optional Sarvam SDK is not installed. "
                    "Run: pip install sarvamai"
                ) from error
            self._client = SarvamAI(api_subscription_key=self.api_key)
        return self._client

    @staticmethod
    def _field(response, name, default=None):
        if isinstance(response, dict):
            return response.get(name, default)
        return getattr(response, name, default)

    def transcribe_file(
        self,
        audio_path: str | Path,
        *,
        language_code: str = "unknown",
        keyterms: list[str] | None = None,
    ) -> dict:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        duration = float(sf.info(str(path)).duration)
        if duration <= 0:
            raise ValueError("Audio file is empty.")
        if duration > self.MAX_CLIP_SECONDS:
            raise ValueError(
                "Hosted Saaras REST transcription supports clips up to "
                f"{self.MAX_CLIP_SECONDS:.0f} seconds; this clip is {duration:.1f}s."
            )

        clean_terms = []
        for term in keyterms or []:
            value = str(term).strip()
            if value and len(value) <= 64 and value not in clean_terms:
                clean_terms.append(value)
        if len(clean_terms) > 50:
            clean_terms = clean_terms[:50]

        started = time.perf_counter()
        with path.open("rb") as audio_file:
            response = self._get_client().speech_to_text.transcribe(
                file=audio_file,
                model=self.MODEL_NAME,
                mode=self.MODE,
                language_code=language_code or "unknown",
                keyterms=clean_terms or None,
            )
        elapsed = round(time.perf_counter() - started, 3)
        transcript = str(self._field(response, "transcript", "") or "").strip()
        if not transcript:
            raise RuntimeError("Saaras v4 returned an empty transcript.")

        detected = self._field(response, "language_code")
        language = (
            str(detected).split("-", 1)[0].lower()
            if detected else "unknown"
        )
        return {
            "transcript": transcript,
            "language": language,
            "language_code": detected,
            "asr_model": self.MODEL_NAME,
            "asr_provider": "sarvam",
            "language_source": "sarvam_saaras_v4",
            "quality_score": None,
            "confidence": None,
            "quality_method": "unavailable_for_saaras",
            "inference_seconds": elapsed,
            "request_id": self._field(response, "request_id"),
        }

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        *,
        language_code: str = "unknown",
        keyterms: list[str] | None = None,
    ) -> dict:
        waveform = np.asarray(audio, dtype=np.float32).squeeze()
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Saaras expects non-empty mono audio.")
        if sample_rate != 16000:
            raise ValueError("Saaras audio must be resampled to 16 kHz.")
        if not np.isfinite(waveform).all():
            raise ValueError("Audio contains non-finite samples.")
        duration = waveform.size / float(sample_rate)
        if duration > self.MAX_CLIP_SECONDS:
            raise ValueError(
                "Hosted Saaras REST transcription supports clips up to "
                f"{self.MAX_CLIP_SECONDS:.0f} seconds; this clip is {duration:.1f}s."
            )

        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(handle.name)
        handle.close()
        try:
            sf.write(str(temp_path), waveform, sample_rate, subtype="PCM_16")
            return self.transcribe_file(
                temp_path,
                language_code=language_code,
                keyterms=keyterms,
            )
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
