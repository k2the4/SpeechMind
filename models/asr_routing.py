"""Small, dependency-light routing signals for the ASR cascade."""
from __future__ import annotations

from models.language_spans import annotate_language_spans


def srota_supports_duration(sample_count: int, sample_rate: int = 16000,
                            max_seconds: float = 30.0) -> bool:
    """Return whether a waveform fits the current Srota worker clip limit."""
    if sample_count <= 0 or sample_rate <= 0:
        return False
    return sample_count / float(sample_rate) <= max_seconds


def whisper_transcript_suggests_hinglish(transcript: str,
                                         primary_language: str = "en") -> bool:
    """Detect transcript evidence that justifies a Srota second pass."""
    annotation = annotate_language_spans(transcript, primary_language)
    return bool(
        annotation["code_switch_detected"]
        or annotation["mixed_script_detected"]
        or annotation["romanized_hindi_detected"]
    )
