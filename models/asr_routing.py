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


def audio_language_suggests_hindi(
    probabilities: dict,
    primary_language: str,
    duration_seconds: float,
    *,
    max_duration_seconds: float = 6.0,
    min_hindi_probability: float = 0.12,
    min_hindi_to_primary_ratio: float = 0.20,
) -> bool:
    """Flag short English-detected audio with a meaningful Hindi probability.

    Whisper's language-token scores are relative model scores, not calibrated
    probabilities. This deliberately conservative secondary signal is only
    used for short clips and never overrides the primary-language decision by
    itself; Srota must independently return Hindi transcript evidence.
    """
    aliases = {"english": "en", "hindi": "hi"}
    primary = aliases.get(str(primary_language or "").lower().strip(),
                          str(primary_language or "").lower().strip())
    if primary != "en" or duration_seconds < 0:
        return False
    if duration_seconds > max_duration_seconds:
        return False
    if not isinstance(probabilities, dict):
        return False

    try:
        hindi_score = float(probabilities.get("hi", probabilities.get("hindi", 0.0)))
        primary_score = float(probabilities.get("en", probabilities.get("english", 0.0)))
    except (TypeError, ValueError):
        return False

    if not (0.0 <= hindi_score <= 1.0 and 0.0 <= primary_score <= 1.0):
        return False
    if hindi_score < min_hindi_probability:
        return False
    return hindi_score >= primary_score * min_hindi_to_primary_ratio



def srota_candidate_valid_for_route(
    transcript: str,
    primary_language: str = "en",
    probe_source: str = "whisper_detection",
) -> bool:
    """Apply route-specific acceptance to a Srota candidate.

    When Hindi is already the primary detected language, Srota must show
    mixed-script or Romanized-Hindi evidence before replacing IndicConformer.
    For an English/uncertain primary route, a Devanagari-only candidate is
    acceptable because it may recover a short Hindi utterance missed by
    Whisper's primary-language decision.
    """
    aliases = {"english": "en", "hindi": "hi"}
    primary = aliases.get(
        str(primary_language or "").lower().strip(),
        str(primary_language or "").lower().strip(),
    )
    if primary == "hi" and probe_source == "whisper_detection":
        return whisper_transcript_suggests_hinglish(transcript, primary)
    return srota_transcript_supports_hindi(transcript, primary)


def srota_transcript_supports_hindi(
    transcript: str,
    primary_language: str = "en",
) -> bool:
    """Accept Srota only when its output contains recognizable Hindi evidence.

    Devanagari-only output is valid here: short Hindi names/phrases may contain
    no English tokens and therefore are not Hinglish code-switches.
    """
    annotation = annotate_language_spans(transcript, primary_language)
    has_hindi_script = (
        annotation["script_evidence"].get("Devanagari", 0) > 0
        and annotation["language_evidence"].get("hi", 0) > 0
    )
    return has_hindi_script or whisper_transcript_suggests_hinglish(
        transcript, primary_language
    )
