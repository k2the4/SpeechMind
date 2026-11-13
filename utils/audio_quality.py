"""Non-destructive audio quality measurements."""
import numpy as np


def analyze_audio_quality(audio, sample_rate: int = 16000) -> dict:
    waveform = np.asarray(audio, dtype=np.float32).squeeze()
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Audio quality analysis expects non-empty mono audio.")
    if sample_rate <= 0 or not np.isfinite(waveform).all():
        raise ValueError("Invalid sample rate or non-finite audio.")
    absolute = np.abs(waveform)
    peak = float(absolute.max())
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    clipped = float(np.mean(absolute >= 0.999))
    silence = float(np.mean(absolute <= 1e-4))
    flags = []
    if peak <= 1e-5 or rms <= 1e-5:
        flags.append("near_silent")
    if clipped > 0.001:
        flags.append("clipping_detected")
    if 0.0 < rms < 0.005 and peak > 1e-5:
        flags.append("very_low_level")
    return {
        "sample_rate_hz": int(sample_rate), "sample_count": int(waveform.size),
        "duration_seconds": round(waveform.size / float(sample_rate), 3),
        "peak_amplitude": round(peak, 6), "rms_amplitude": round(rms, 6),
        "clipped_sample_ratio": round(clipped, 6),
        "near_silence_sample_ratio": round(silence, 6),
        "status": "unusable" if "near_silent" in flags else ("review" if flags else "good"),
        "flags": flags, "is_probability": False,
        "processing": "measurement_only_no_trim_no_denoise_no_gain_change",
    }
