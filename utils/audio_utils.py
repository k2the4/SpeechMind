"""
SpeechMind - Audio preprocessing utilities.

Shared by the ASR branch and the voice-emotion branch so both
always see the same resampled, trimmed, normalized audio.
"""
import numpy as np
import librosa

TARGET_SAMPLE_RATE = 16000


def load_audio(path: str, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load an audio file, resample to 16kHz mono, trim leading/trailing silence."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    return audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1] so loudness is consistent across recordings/mics."""
    peak = np.max(np.abs(audio)) if len(audio) else 0
    if peak > 0:
        audio = audio / peak
    return audio


def preprocess(path: str) -> np.ndarray:
    """Full preprocessing pipeline used by pipeline.py before ASR/voice-emotion."""
    audio = load_audio(path)
    audio = normalize_audio(audio)
    return audio
