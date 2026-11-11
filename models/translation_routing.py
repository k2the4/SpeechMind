"""Language-aware routing for English-meaning generation.

This module keeps language/script detection separate from translation itself.
It never mutates the original transcript.
"""
from __future__ import annotations

from models.translation import FLORES_CODE


def resolve_translation_route(
    detected_language: str | None,
    language_spans: list[dict] | None = None,
    romanized_hindi_detected: bool = False,
) -> dict:
    """Resolve whether and how an ASR transcript can be translated to English.

    IndicTrans2 is monolingual-source Indic->English. For mixed Devanagari
    Hindi + Latin English, use Hindi as the source language as a best-effort
    route. Romanized Hindi requires a transliteration stage and is deliberately
    not passed directly to IndicTrans2.
    """
    detected = str(detected_language or "unknown").lower().strip()
    aliases = {"english": "en", "hindi": "hi"}
    detected = aliases.get(detected, detected)

    spans = language_spans if isinstance(language_spans, list) else []
    languages = {
        str(span.get("language") or "").lower().strip()
        for span in spans
        if isinstance(span, dict)
    }
    languages.discard("")
    romanized_hindi = bool(romanized_hindi_detected) or any(
        str(span.get("language") or "").lower().strip() == "hi"
        and str(span.get("script") or "").lower().strip() == "latin"
        for span in spans
        if isinstance(span, dict)
    )

    source_language = detected
    # Romanized-Hindi evidence must take precedence over an English primary
    # label; otherwise the identity route would return untranslated Hindi.
    if romanized_hindi and detected in {"en", "unknown"}:
        source_language = "hi"
    # A primary English label must not hide clear Hindi spans in a mixed
    # transcript. Hindi is the IndicTrans2 source language for this route.
    elif "hi" in languages and detected in {"en", "unknown"}:
        source_language = "hi"
    elif source_language == "unknown" and "hi" in languages:
        source_language = "hi"

    if source_language in {"en", "english"}:
        return {
            "available": True,
            "source_language": "en",
            "method": "original_english_transcript",
            "error": None,
            "note": None,
            "languages_in_transcript": sorted(languages or {"en"}),
            "romanized_hindi_detected": romanized_hindi,
        }

    if source_language == "hi" and romanized_hindi:
        return {
            "available": False,
            "source_language": "hi",
            "method": "romanized_hindi_transliteration_required",
            "error": (
                "Romanized Hindi was detected. It has not been sent directly "
                "to IndicTrans2 because the required Hindi transliteration "
                "stage is not installed and validated yet."
            ),
            "note": None,
            "languages_in_transcript": sorted(languages),
            "romanized_hindi_detected": True,
        }

    flores_code = FLORES_CODE.get(source_language)
    if flores_code is None:
        return {
            "available": False,
            "source_language": source_language,
            "method": "unsupported_source_language",
            "error": (
                "English meaning is not configured for detected language: "
                f"{source_language or 'unknown'}."
            ),
            "note": None,
            "languages_in_transcript": sorted(languages),
            "romanized_hindi_detected": romanized_hindi,
        }

    is_mixed_hindi_english = (
        source_language == "hi" and {"hi", "en"}.issubset(languages)
    )
    return {
        "available": True,
        "source_language": source_language,
        "method": (
            "indictrans2_mixed_script_best_effort"
            if is_mixed_hindi_english else "indictrans2"
        ),
        "error": None,
        "note": (
            "Hindi-English mixed-script translation is best-effort and has "
            "not yet been evaluated on a held-out Hinglish meaning set."
            if is_mixed_hindi_english else None
        ),
        "languages_in_transcript": sorted(languages),
        "romanized_hindi_detected": romanized_hindi,
    }
