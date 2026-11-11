"""Small, dependency-light routing signals for the ASR cascade."""
from __future__ import annotations

from models.language_spans import (
    annotate_language_spans,
    STRONG_ROMANIZED_HINDI_WORDS,
)


# These words are specific enough to justify a second pass even when they are
# the only Romanized-Hindi token in a short transcript.
def srota_supports_duration(sample_count: int, sample_rate: int = 16000,
                            max_seconds: float = 30.0) -> bool:
    """Return whether a waveform fits the current Srota worker clip limit."""
    if sample_count <= 0 or sample_rate <= 0 or max_seconds <= 0:
        return False
    return sample_count / float(sample_rate) <= max_seconds


def whisper_transcript_suggests_hinglish(transcript: str,
                                         primary_language: str = "en") -> bool:
    """Require meaningful transcript evidence before invoking Srota.

    Mixed Devanagari + recognized English is direct code-switch evidence.
    Latin-only output needs either two distinct Romanized-Hindi lexicon tokens
    or one higher-specificity Hindi token. This prevents one ambiguous English
    word such as "main" from triggering the slower Hinglish model.
    """
    annotation = annotate_language_spans(transcript, primary_language)

    if annotation["mixed_script_detected"]:
        return True

    romanized_tokens = {
        token.casefold().replace("’", "'")
        for span in annotation["language_spans"]
        if span["script"] == "Latin" and span["language"] == "hi"
        for token in span["tokens"]
    }
    return (
        len(romanized_tokens) >= 2
        or bool(romanized_tokens & STRONG_ROMANIZED_HINDI_WORDS)
    )
