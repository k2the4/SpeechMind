"""Small dependency-free ASR text normalization and error-rate metrics."""
from __future__ import annotations

import unicodedata


def normalize_asr_text(text: str, *, remove_punctuation: bool = True) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    if remove_punctuation:
        value = "".join(
            char for char in value
            if not unicodedata.category(char).startswith("P")
        )
    return " ".join(value.split())


def edit_distance(reference, hypothesis) -> int:
    """Levenshtein distance over a sequence of tokens or characters."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (ref_item != hyp_item)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def error_counts(reference: str, hypothesis: str) -> dict:
    ref = normalize_asr_text(reference)
    hyp = normalize_asr_text(hypothesis)
    ref_words = ref.split()
    hyp_words = hyp.split()
    word_errors = edit_distance(ref_words, hyp_words)
    char_errors = edit_distance(list(ref), list(hyp))
    return {
        "word_errors": word_errors,
        "reference_words": len(ref_words),
        "character_errors": char_errors,
        "reference_characters": len(ref),
        "wer": word_errors / max(1, len(ref_words)),
        "cer": char_errors / max(1, len(ref)),
    }


def aggregate_error_counts(items: list[dict]) -> dict:
    word_errors = sum(int(item["word_errors"]) for item in items)
    reference_words = sum(int(item["reference_words"]) for item in items)
    char_errors = sum(int(item["character_errors"]) for item in items)
    reference_characters = sum(int(item["reference_characters"]) for item in items)
    return {
        "samples": len(items),
        "word_errors": word_errors,
        "reference_words": reference_words,
        "character_errors": char_errors,
        "reference_characters": reference_characters,
        "wer": word_errors / max(1, reference_words),
        "cer": char_errors / max(1, reference_characters),
    }
