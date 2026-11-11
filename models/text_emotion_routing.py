"""Conservative language gate for semantic emotion inference.

A single utterance-level language guess is not sufficient for code-switched
text. Mixed-language text is only sent to a model that explicitly declares
Hinglish support in its checkpoint metadata.
"""


def resolve_text_emotion_language(
    detected_language,
    languages_present=None,
    code_switch_detected=False,
    mixed_script_detected=False,
    romanized_hindi_detected=False,
    supported_languages=None,
):
    """Return a safe language route for the text-emotion model.

    Pure English and pure supported-language transcripts retain their normal
    routes. Hindi-English, mixed-script, and Romanized-Hindi transcripts require
    an explicit Hinglish capability declaration. Other mixed-language
    transcripts are withheld rather than sent to an unvalidated monolingual
    classifier.
    """
    detected = str(detected_language or "unknown").lower().strip()
    aliases = {"english": "en", "hindi": "hi"}
    detected = aliases.get(detected, detected)
    languages = {
        str(language).lower().strip()
        for language in (languages_present or [])
        if str(language).strip()
    }
    supported = {
        str(language).lower().strip()
        for language in (supported_languages or [])
        if str(language).strip()
    }

    is_hinglish = (
        bool(romanized_hindi_detected)
        or {"hi", "en"}.issubset(languages)
        or (
            bool(mixed_script_detected)
            and (detected in {"hi", "en"} or "hi" in languages)
        )
        or (
            bool(code_switch_detected)
            and (detected in {"hi", "en"} or "hi" in languages)
        )
    )

    if is_hinglish:
        if "hinglish" not in supported:
            return {
                "available": False,
                "language": "hinglish",
                "reason": (
                    "This transcript appears to contain Hinglish or Romanized "
                    "Hindi, but the loaded text-emotion checkpoint does not "
                    "explicitly declare Hinglish support. Text emotion was "
                    "withheld rather than applying an English-only or "
                    "Hindi-only model to mixed-language text."
                ),
            }
        return {
            "available": True,
            "language": "hinglish",
            "reason": None,
        }

    is_mixed = bool(code_switch_detected or mixed_script_detected or len(languages) > 1)
    if is_mixed:
        return {
            "available": False,
            "language": "unsupported_mixed",
            "reason": (
                "The transcript contains multiple languages or scripts, but "
                "the loaded text-emotion model has no validated route for this "
                "combination. Text emotion was withheld."
            ),
        }

    if detected == "unknown" and len(languages) == 1:
        detected = next(iter(languages))

    return {
        "available": True,
        "language": detected,
        "reason": None,
    }
