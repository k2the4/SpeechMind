"""
Shared audio preprocessing for ASR and voice emotion recognition.
All model inputs are mono float32 waveforms sampled at 16 kHz.
"""
import numpy as np

# Configure FFmpeg before importing/using decoders such as librosa.
from utils.ffmpeg_setup import FFMPEG_BIN  # noqa: F401
import librosa

TARGET_SAMPLE_RATE = 16000


def load_audio(path: str, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"No audio samples could be decoded from: {path}")
    audio, _ = librosa.effects.trim(audio, top_db=25)
    return audio.astype(np.float32, copy=False)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Audio is empty after preprocessing.")
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def preprocess(path: str) -> np.ndarray:
    return normalize_audio(load_audio(path))
