"""Bodhan API client for Hinglish-first speech recognition and translation.

Audio is sent to Bodhan only when BODHAN_API_KEY is configured. Without a key,
SpeechMind stays on its local Whisper fallback. The API accepts at most 30
seconds per transcription request; longer clips are split into overlapping
27-second chunks and boundary text is de-duplicated.
"""
from __future__ import annotations

import io
import os
import re
import unicodedata
from typing import Optional

import numpy as np
import requests
import soundfile as sf
from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://api.bodhan.ai"
SAMPLE_RATE = 16000
MAX_REQUEST_SECONDS = 27.0
CHUNK_OVERLAP_SECONDS = 1.0


class BodhanAPIError(RuntimeError):
    """Raised when the configured Bodhan endpoint cannot complete a request."""


class BodhanClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: int = 90):
        if not api_key or not api_key.strip():
            raise ValueError("A non-empty Bodhan API key is required.")
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.session = requests.Session()

    @classmethod
    def from_env(cls) -> Optional["BodhanClient"]:
        """Load an optional local .env file and return a client when configured."""
        load_dotenv(override=False)
        api_key = os.getenv("BODHAN_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.getenv("BODHAN_API_BASE_URL", DEFAULT_BASE_URL).strip()
        try:
            timeout = int(os.getenv("BODHAN_API_TIMEOUT", "90"))
        except ValueError:
            timeout = 90
        return cls(api_key=api_key, base_url=base_url or DEFAULT_BASE_URL, timeout=timeout)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def language_profile(text: str) -> dict:
        """Describe observable scripts without pretending Latin Hinglish is English."""
        script_ranges = {
            "hi_or_mr": (0x0900, 0x097F),  # Devanagari cannot distinguish Hindi/Marathi alone
            "bn": (0x0980, 0x09FF),
            "pa": (0x0A00, 0x0A7F),
            "ta": (0x0B80, 0x0BFF),
            "te": (0x0C00, 0x0C7F),
        }
        scripts = set()
        for char in text:
            code = ord(char)
            if char.isascii() and char.isalpha():
                scripts.add("latin")
                continue
            for name, (start, end) in script_ranges.items():
                if start <= code <= end:
                    scripts.add(name)
                    break

        if not scripts:
            profile = "undetermined"
            languages = []
        elif scripts == {"latin"}:
            profile = "english_or_romanized_hinglish"
            languages = ["en", "hi/romanized"]
        elif "latin" in scripts:
            profile = "code_mixed_or_mixed_script"
            languages = sorted(scripts)
        elif len(scripts) == 1:
            script = next(iter(scripts))
            profile = script
            languages = [script]
        else:
            profile = "multiple_indic_scripts"
            languages = sorted(scripts)

        return {
            "language_profile": profile,
            "script_evidence": sorted(scripts),
            "language_profile_note": (
                "Script-based estimate only. Latin-script English and romanized "
                "Hinglish cannot be reliably separated from the transcript alone."
                if profile == "english_or_romanized_hinglish"
                else None
            ),
        }

    @staticmethod
    def _audio_chunks(audio: np.ndarray) -> list[np.ndarray]:
        waveform = np.asarray(audio, dtype=np.float32).squeeze()
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Bodhan transcription expects non-empty mono audio.")
        if not np.isfinite(waveform).all():
            raise ValueError("Audio contains NaN or infinite samples.")
        max_samples = int(MAX_REQUEST_SECONDS * SAMPLE_RATE)
        if waveform.size <= max_samples:
            return [waveform]

        step = int((MAX_REQUEST_SECONDS - CHUNK_OVERLAP_SECONDS) * SAMPLE_RATE)
        chunks = []
        start = 0
        while start < waveform.size:
            end = min(start + max_samples, waveform.size)
            chunk = waveform[start:end]
            if chunk.size:
                chunks.append(chunk)
            if end >= waveform.size:
                break
            start += step
        return chunks

    @staticmethod
    def _wav_bytes(audio: np.ndarray) -> bytes:
        output = io.BytesIO()
        sf.write(output, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return output.getvalue()

    @staticmethod
    def _merge_chunk_text(previous: str, current: str) -> str:
        """Remove repeated words caused by the one-second overlapping audio window."""
        previous = (previous or "").strip()
        current = (current or "").strip()
        if not previous:
            return current
        if not current:
            return previous

        left = previous.split()
        right = current.split()
        normal = lambda token: re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", token)).casefold()
        max_overlap = min(14, len(left), len(right))
        overlap = 0
        for size in range(max_overlap, 0, -1):
            if [normal(x) for x in left[-size:]] == [normal(x) for x in right[:size]]:
                overlap = size
                break
        suffix = " ".join(right[overlap:])
        return (previous + (" " + suffix if suffix else "")).strip()

    def _post_transcription(self, chunk: np.ndarray) -> str:
        url = f"{self.base_url}/v1/audio/transcriptions"
        files = {
            "file": ("speechmind_audio.wav", self._wav_bytes(chunk), "audio/wav")
        }
        data = {"model": "indic-transcribe"}
        try:
            response = self.session.post(
                url, headers=self._headers(), files=files, data=data,
                timeout=self.timeout,
            )
        except requests.RequestException as error:
            raise BodhanAPIError(f"Bodhan transcription request failed: {error}") from error

        if response.status_code >= 400:
            detail = response.text[:500]
            raise BodhanAPIError(
                f"Bodhan transcription returned HTTP {response.status_code}: {detail}"
            )
        try:
            body = response.json()
        except ValueError as error:
            raise BodhanAPIError("Bodhan transcription returned invalid JSON.") from error
        transcript = body.get("text")
        if not isinstance(transcript, str):
            raise BodhanAPIError("Bodhan transcription response did not contain a text field.")
        return transcript.strip()

    def transcribe(self, audio: np.ndarray) -> dict:
        chunks = self._audio_chunks(audio)
        transcript = ""
        for chunk in chunks:
            part = self._post_transcription(chunk)
            transcript = self._merge_chunk_text(transcript, part)

        profile = self.language_profile(transcript)
        return {
            "transcript": transcript,
            "language": profile["language_profile"],
            "language_profile": profile["language_profile"],
            "language_profile_note": profile["language_profile_note"],
            "script_evidence": profile["script_evidence"],
            "language_detection_score": None,
            "language_source": "bodhan_auto_transcription_script_heuristic",
            "confidence": None,
            "quality_score": None,
            "quality_method": "not_provided_by_bodhan_api",
            "asr_model": "bodhan-indic-transcribe",
            "asr_provider": "bodhan_api",
            "asr_chunks": len(chunks),
        }

    def translate_to_english(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "translation": text,
                "target_language": "en",
                "truncated": False,
                "provider": "none",
            }
        url = f"{self.base_url}/translate"
        payload = {"text": text, "target_language": "en"}
        try:
            response = self.session.post(
                url,
                headers={**self._headers(), "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as error:
            raise BodhanAPIError(f"Bodhan translation request failed: {error}") from error

        if response.status_code >= 400:
            detail = response.text[:500]
            raise BodhanAPIError(
                f"Bodhan translation returned HTTP {response.status_code}: {detail}"
            )
        try:
            body = response.json()
        except ValueError as error:
            raise BodhanAPIError("Bodhan translation returned invalid JSON.") from error
        if body.get("error"):
            error = body["error"]
            raise BodhanAPIError(
                f"Bodhan translation failed: {error.get('code', 'unknown_error')}"
            )
        translation = body.get("translation")
        if not isinstance(translation, str):
            raise BodhanAPIError("Bodhan translation response had no translation field.")
        if body.get("truncated"):
            raise BodhanAPIError("Bodhan marked the English translation as truncated.")
        return {
            "translation": translation.strip(),
            "target_language": body.get("target_language", "en"),
            "truncated": False,
            "provider": "bodhan_indic_translate",
            "usage": body.get("usage"),
        }
