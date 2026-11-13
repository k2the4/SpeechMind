"""
Shared audio preprocessing for ASR and voice emotion recognition.
All model inputs are mono float32 waveforms sampled at 16 kHz.
"""
from __future__ import annotations

import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from utils.audio_quality import analyze_audio_quality

# Configure FFmpeg before importing/using the compatibility decoder.
from utils.ffmpeg_setup import FFMPEG_BIN  # noqa: F401
import librosa

TARGET_SAMPLE_RATE = 16000


def _load_with_soundfile(path: str, sr: int) -> np.ndarray | None:
    """Fast path for WAV/FLAC/OGG handled directly by libsndfile."""
    try:
        audio, source_rate = sf.read(
            path, dtype="float32", always_2d=True
        )
    except (RuntimeError, OSError, ValueError):
        return None

    if audio.size == 0:
        raise ValueError(f"No audio samples could be decoded from: {path}")
    # Arithmetic averaging is deterministic and avoids a librosa resample path.
    mono = audio.mean(axis=1, dtype=np.float32)
    if int(source_rate) != int(sr):
        divisor = math.gcd(int(source_rate), int(sr))
        mono = resample_poly(
            mono, int(sr) // divisor, int(source_rate) // divisor
        ).astype(np.float32, copy=False)
    return np.asarray(mono, dtype=np.float32)


def load_audio(path: str, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode mono audio; use libsndfile first and librosa for fallback formats."""
    audio = _load_with_soundfile(path, sr)
    if audio is None:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"No audio samples could be decoded from: {path}")
    # Preserve leading/trailing pauses and quiet speech. Quality is measured
    # independently; destructive trimming can remove vocal/emotional cues.
    return audio.astype(np.float32, copy=False)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Audio is empty after preprocessing.")
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def preprocess_with_quality(path: str):
    """Decode once, measure the original waveform, then normalize for model input."""
    audio = load_audio(path)
    quality = analyze_audio_quality(audio, sample_rate=TARGET_SAMPLE_RATE)
    return normalize_audio(audio), quality


def preprocess(path: str) -> np.ndarray:
    """Backward-compatible model input helper."""
    audio, _ = preprocess_with_quality(path)
    return audio
