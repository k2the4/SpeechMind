"""
SpeechMind automatic multilingual ASR.

The user never selects a language. The local Whisper path transcribes in automatic
multilingual mode. An optional Bodhan provider is available for code-switch-aware
Indian ASR, but it is only used when explicitly selected in the local .env file.
"""
from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
import torch
import whisper

from models.asr_quality import estimate_whisper_quality
from models.language_utils import infer_transcript_language

WHISPER_MODEL = os.getenv("SPEECHMIND_WHISPER_MODEL", "small")
BODHAN_BASE_URL = os.getenv("BODHAN_API_BASE_URL", "https://api.bodhan.ai/v1").rstrip("/")
BODHAN_ASR_MODEL = os.getenv("BODHAN_ASR_MODEL", "indic-transcribe")
SAMPLE_RATE = 16000
BODHAN_MAX_SECONDS = 28.0


def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= char <= "\u097f" for char in text)


def _has_latin(text: str) -> bool:
    return any(("a" <= char.lower() <= "z") for char in text)


def _transcript_language_label(text: str, provider_language: str | None) -> str:
    """Use script evidence for mixed output; otherwise preserve provider metadata."""
    if _has_devanagari(text) and _has_latin(text):
        return "hi-en-mixed"
    if provider_language:
        normalized = str(provider_language).lower().strip()
        return "en" if normalized in {"en", "english"} else normalized
    return "unknown"


def _chunk_for_hosted_asr(audio: np.ndarray, max_seconds: float = BODHAN_MAX_SECONDS):
    """Split long audio near silence; do not modify the original voice-branch audio."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    max_samples = int(max_seconds * SAMPLE_RATE)
    if len(audio) <= max_samples:
        return [(0, len(audio), audio)]

    try:
        import librosa

        intervals = librosa.effects.split(
            audio, top_db=32, frame_length=512, hop_length=128
        )
    except Exception:
        intervals = np.asarray([], dtype=np.int64)

    if len(intervals) == 0:
        intervals = np.asarray(
            [(start, min(start + max_samples, len(audio)))
             for start in range(0, len(audio), max_samples)],
            dtype=np.int64,
        )

    chunks = []
    pending_start = None
    pending_end = None

    for raw_start, raw_end in intervals:
        start, end = int(raw_start), int(raw_end)
        while end - start > max_samples:
            if pending_start is not None and pending_end is not None:
                chunks.append((pending_start, pending_end, audio[pending_start:pending_end]))
                pending_start = pending_end = None
            cut = start + max_samples
            chunks.append((start, cut, audio[start:cut]))
            start = cut

        if pending_start is None:
            pending_start, pending_end = start, end
        elif end - pending_start <= max_samples:
            pending_end = end
        else:
            chunks.append((pending_start, pending_end, audio[pending_start:pending_end]))
            pending_start, pending_end = start, end

    if pending_start is not None and pending_end is not None:
        chunks.append((pending_start, pending_end, audio[pending_start:pending_end]))

    return [item for item in chunks if len(item[2]) > int(0.15 * SAMPLE_RATE)]


def _deduplicate_chunk_join(texts: list[str]) -> str:
    """Remove a short exact repeated word overlap, if the provider emits one."""
    merged: list[str] = []
    for text in texts:
        words = str(text or "").strip().split()
        if not words:
            continue
        overlap = 0
        max_overlap = min(8, len(merged), len(words))
        for size in range(max_overlap, 0, -1):
            if [w.casefold() for w in merged[-size:]] == [
                w.casefold() for w in words[:size]
            ]:
                overlap = size
                break
        merged.extend(words[overlap:])
    return " ".join(merged).strip()


class BodhanASRClient:
    """Hosted code-switch-aware Indian ASR. Audio leaves the device only in this mode."""

    def __init__(self):
        self.api_key = (
            os.getenv("BODHAN_ASR_API_KEY")
            or os.getenv("BODHAN_API_KEY")
            or ""
        ).strip()
        if not self.api_key:
            raise RuntimeError(
                "Bodhan ASR was selected but BODHAN_ASR_API_KEY is missing. "
                "Add the key to your local .env file; do not commit it."
            )
        try:
            import requests
        except ImportError as error:
            raise RuntimeError(
                "Bodhan ASR requires the 'requests' package. Install requirements.txt."
            ) from error
        self.requests = requests
        self.endpoint = f"{BODHAN_BASE_URL}/audio/transcriptions"

    def _transcribe_chunk(self, audio: np.ndarray) -> dict[str, Any]:
        try:
            import soundfile as sf
        except ImportError as error:
            raise RuntimeError("Bodhan ASR requires soundfile.") from error

        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        try:
            response = self.requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": ("speechmind.wav", buffer, "audio/wav")},
                data={"model": BODHAN_ASR_MODEL},
                timeout=(15, 180),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            detail = getattr(getattr(error, "response", None), "text", None)
            raise RuntimeError(
                f"Bodhan transcription request failed: {detail or error}"
            ) from error

        if isinstance(payload, str):
            return {"text": payload}
        if not isinstance(payload, dict):
            raise RuntimeError("Bodhan ASR returned an unsupported response format.")
        if payload.get("error"):
            raise RuntimeError(f"Bodhan ASR error: {payload['error']}")

        transcript = (
            payload.get("text")
            or payload.get("transcript")
            or payload.get("transcription")
        )
        if not isinstance(transcript, str):
            raise RuntimeError(
                "Bodhan ASR response did not contain a text/transcript field."
            )
        return {
            "text": transcript.strip(),
            "language": payload.get("language"),
            "segments": payload.get("segments"),
        }

    def transcribe(self, audio: np.ndarray) -> dict:
        chunks = _chunk_for_hosted_asr(audio)
        results = [self._transcribe_chunk(chunk) for _, _, chunk in chunks]
        transcript = _deduplicate_chunk_join([item["text"] for item in results])
        provider_languages = [
            str(item["language"]).lower().strip()
            for item in results if item.get("language")
        ]
        unique_languages = sorted(set(provider_languages))
        language = (
            unique_languages[0] if len(unique_languages) == 1
            else "mixed" if len(unique_languages) > 1
            else None
        )
        language = infer_transcript_language(transcript, language, unique_languages)
        return {
            "transcript": transcript,
            "language": language,
            "languages_detected": unique_languages,
            "confidence": None,
            "quality_score": None,
            "quality_method": "not_exposed_by_bodhan_response",
            "language_detection_score": None,
            "language_source": "bodhan_asr_provider" if language != "unknown" else "provider_unspecified",
            "asr_model": BODHAN_ASR_MODEL,
            "asr_backend": "bodhan",
            "asr_chunk_count": len(chunks),
            "asr_segments": [
                {
                    "start_sec": round(start / SAMPLE_RATE, 3),
                    "end_sec": round(end / SAMPLE_RATE, 3),
                    "text": item["text"],
                }
                for (start, end, _), item in zip(chunks, results)
            ],
        }


class SpeechRecognizer:
    def __init__(self):
        self.backend = os.getenv("SPEECHMIND_ASR_BACKEND", "whisper").strip().lower()
        if self.backend not in {"whisper", "bodhan"}:
            raise ValueError(
                "SPEECHMIND_ASR_BACKEND must be 'whisper' or 'bodhan'. "
                "Experimental local ASR candidates are benchmarked separately."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper = None
        self.bodhan = None

        if self.backend == "bodhan":
            self.bodhan = BodhanASRClient()
            self.model_name = BODHAN_ASR_MODEL
            print("[ASR] Bodhan code-switch-aware hosted ASR selected.")
        else:
            self.model_name = f"openai/whisper-{WHISPER_MODEL}"
            print(f"[ASR] Loading Whisper-{WHISPER_MODEL} in automatic-language mode...")
            self.whisper = whisper.load_model(WHISPER_MODEL, device=str(self.device))
            print(f"[ASR] Whisper loaded on {self.device}; no language is forced.")

    def transcribe(self, audio) -> dict:
        if isinstance(audio, torch.Tensor):
            waveform = audio.detach().cpu().numpy()
        else:
            waveform = audio
        waveform = np.asarray(waveform, dtype=np.float32).squeeze()
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("ASR expects non-empty mono audio.")
        if not np.isfinite(waveform).all():
            raise ValueError("ASR received NaN or infinite audio.")

        if self.backend == "bodhan":
            return self.bodhan.transcribe(waveform)

        # Whisper's transcription pass performs its own automatic language
        # detection. We intentionally do not run a separate file-level language
        # classifier and route the entire recording to one monolingual decoder.
        result = self.whisper.transcribe(
            waveform,
            task="transcribe",
            language=None,
            fp16=False,
            verbose=False,
            beam_size=5,
            condition_on_previous_text=False,
        )
        transcript = str(result.get("text", "")).strip()
        detected = str(result.get("language") or "unknown").lower().strip()
        language = infer_transcript_language(transcript, detected)
        quality = estimate_whisper_quality(result.get("segments", []), transcript)

        return {
            "transcript": transcript,
            "language": language,
            "languages_detected": [language] if language != "unknown" else [],
            "confidence": quality["quality_score"],
            "quality_score": quality["quality_score"],
            "quality_method": "whisper_token_quality_heuristic_uncalibrated",
            "avg_logprob": quality["avg_logprob"],
            "avg_no_speech_prob": quality["avg_no_speech_prob"],
            "avg_compression_ratio": quality["avg_compression_ratio"],
            "language_detection_score": None,
            "language_source": "whisper_transcription_auto",
            "asr_model": "whisper-small",
            "asr_backend": "whisper",
            "asr_chunk_count": 1,
            "asr_segments": result.get("segments", []),
        }
