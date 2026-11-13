"""Conservative transcript-level language/script span annotation.

Spans use Python character offsets into the untouched transcript. They are not
audio timestamps. Latin words are labelled English or Romanized Hindi only
when supported by small high-precision lexicons; ambiguous words stay unknown.
"""
from __future__ import annotations

import unicodedata
from collections import Counter

def _is_letter_number_or_mark(character: str) -> bool:
    """Include Unicode combining marks, including Devanagari matras."""
    return unicodedata.category(character)[0] in {"L", "N", "M"}


def _iter_word_spans(text: str):
    """Yield exact [start, end) word offsets without dropping combining marks."""
    index = 0
    length = len(text)
    while index < length:
        if unicodedata.category(text[index])[0] not in {"L", "N"}:
            index += 1
            continue

        start = index
        index += 1
        while index < length:
            character = text[index]
            if _is_letter_number_or_mark(character):
                index += 1
                continue
            if (
                character in {"'", "’", "\u200c", "\u200d"}
                and index + 1 < length
                and unicodedata.category(text[index + 1])[0] in {"L", "N"}
            ):
                index += 1
                continue
            break

        yield start, index



SUPPORTED_LANGUAGES = {
    "as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "ta", "te", "ur"
}

# High-precision conversational Romanized-Hindi indicators. Deliberately avoids
# ambiguous short forms such as "is", "to", "no", and "me".
ROMANIZED_HINDI_WORDS = {
    "aaj", "aap", "aapka", "aapki", "aapko", "ab", "abhi", "accha", "achha",
    "acha", "acchi", "achhi", "ache", "achhe", "agar", "aisa", "aise", "aisi",
    "aur", "bahut", "bohot", "batao", "bataiye", "baat", "bhi", "bhai", "bol",
    "bola", "boli", "bolo", "chahta", "chahti", "chahiye", "chaiye", "chalo",
    "dikhao", "dijiye", "dost", "ga", "gaya", "gayi", "gaye", "ghar", "ham",
    "hamara", "hamari", "hamare", "haan", "han", "hai", "hain", "ho", "hoga",
    "hogi", "honge", "hona", "hoon", "hun", "hu", "hum", "idhar", "iska", "iske",
    "iski", "jaisa", "jaise", "jaisi", "ja", "jana", "jaana", "jao", "ji", "kahan",
    "kaha", "kaisa", "kaise", "kaisi", "kar", "kare", "karen", "karke", "karna",
    "karne", "karta", "karte", "karti", "karunga", "karungi", "ka", "kaafi",
    "kaam", "kal", "kam", "karo", "karun", "karu", "karna", "ke", "kitna",
    "kitne", "kitni", "ko", "kyun", "kyu", "kya", "kyunki", "lekin", "liye",
    "log", "main", "mai", "mein", "mera", "mere", "meri", "mujhe", "mujhko",
    "mat", "matlab", "mil", "mila", "mile", "milega", "milega", "mujhse",
    "naam", "nahi", "nahin", "nhi", "nikal", "par", "pe", "pata",
    "raha", "rahe", "rahi", "rha", "rhe", "rhi", "sab", "sahi", "samajh",
    "samjha", "samjhi", "se", "sirf", "thoda", "thodi", "thode", "tha", "the",
    "thi", "toh", "tum", "tumhara", "tumhari", "tumhare", "tumhe", "tujhe",
    "tera", "tere", "teri", "udhar", "unka", "unke", "unki", "uska", "uske",
    "uski", "waise", "wala", "wali", "wale", "wahan", "yaar", "ye", "yeh",
    "yahan", "yahi", "ya", "woh", "wo"
}

# Context-sensitive Romanized-Hindi tokens. These are valid Hindi forms but
# common enough in English ("the", "main") that they must not establish Hindi
# evidence by themselves when Whisper's primary-language guess is English.
AMBIGUOUS_ROMANIZED_HINDI_WORDS = {
    "main", "mai", "the", "ka", "ke", "ko", "se", "par", "pe",
    "ja", "ya", "toh", "wo", "ye", "is", "me", "kal", "kam", "ga", "ji",
}
STRONG_ROMANIZED_HINDI_WORDS = {
    "aaj", "aapko", "achha", "acha", "bahut", "batao", "chahiye",
    "hai", "hain", "hoon", "hun", "kyunki", "mujhe", "mujhko",
    "mujhse", "nahi", "nahin", "nhi", "raha", "rahe", "rahi",
    "samajh", "tumhe", "tujhe",
}

# A compact list of common English function words and frequent Hinglish-domain
# words. Unknown Latin tokens are not forced into English.
ENGLISH_WORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "because", "been", "before", "being", "between", "but",
    "by", "can", "color", "colour", "could", "day", "did", "do", "does", "down",
    "each", "even", "every", "favourite", "favorite", "feel", "for", "from",
    "get", "give", "go", "good", "great", "had", "has", "have", "he", "her",
    "here", "how", "i", "if", "image", "in", "into", "is", "it", "its", "just",
    "know", "like", "look", "lot", "make", "many", "may", "me", "meeting",
    "more", "most", "much", "my", "name", "need", "new", "not", "now", "of",
    "off", "on", "one", "only", "or", "other", "our", "out", "over", "people",
    "please", "project", "really", "say", "see", "she", "should", "so", "some",
    "something", "story", "take", "than", "that", "the", "their", "them", "then",
    "design", "designer", "layout", "sure", "interface", "figma", "python",
    "there", "these", "they", "thing", "think", "this", "those", "thought", "time", "to",
    "today", "too", "up", "us", "use", "very", "was", "we", "were", "what",
    "when", "where", "which", "who", "will", "with", "work", "would", "yeah",
    "yes", "you", "your"
}

SCRIPT_MARKERS = (
    ("DEVANAGARI", "Devanagari"),
    ("LATIN", "Latin"),
    ("BENGALI", "Bengali"),
    ("GURMUKHI", "Gurmukhi"),
    ("GUJARATI", "Gujarati"),
    ("TAMIL", "Tamil"),
    ("TELUGU", "Telugu"),
    ("KANNADA", "Kannada"),
    ("MALAYALAM", "Malayalam"),
    ("ARABIC", "Arabic"),
)


def _script_for_token(token: str) -> str:
    scripts = set()
    for character in token:
        name = unicodedata.name(character, "")
        for marker, label in SCRIPT_MARKERS:
            if marker in name:
                scripts.add(label)
                break
    if not scripts:
        return "Other"
    if len(scripts) == 1:
        return next(iter(scripts))
    return "Mixed"


def _primary_language(value: str | None) -> str:
    language = str(value or "unknown").lower().strip()
    aliases = {"english": "en", "hindi": "hi"}
    language = aliases.get(language, language)
    return language if language in SUPPORTED_LANGUAGES or language == "en" else "unknown"


def _classify_token(
    token: str,
    script: str,
    primary_language: str,
    allow_primary_hint: bool = True,
) -> tuple[str, float, str]:
    normalized = token.casefold().replace("’", "'")
    if script == "Devanagari":
        language = primary_language if primary_language in SUPPORTED_LANGUAGES else "hi"
        return language, 0.96, "unicode_script"

    if script == "Latin":
        if normalized in ROMANIZED_HINDI_WORDS and (
            normalized not in AMBIGUOUS_ROMANIZED_HINDI_WORDS
            or not allow_primary_hint
        ):
            return "hi", 0.90, "romanized_hindi_lexicon"
        if normalized in ENGLISH_WORDS:
            return "en", 0.86, "common_english_lexicon"
        if primary_language == "en" and allow_primary_hint:
            return "en", 0.62, "primary_language_hint"
        return "und", 0.25, "ambiguous_latin_token"

    if script in {"Bengali", "Gurmukhi", "Gujarati", "Tamil", "Telugu", "Kannada", "Malayalam", "Arabic"}:
        script_language = {
            "Bengali": "bn", "Gurmukhi": "pa", "Gujarati": "gu",
            "Tamil": "ta", "Telugu": "te", "Kannada": "kn",
            "Malayalam": "ml", "Arabic": primary_language if primary_language == "ur" else "und",
        }.get(script, "und")
        return script_language, 0.92, "unicode_script"

    return "und", 0.0, "unknown_script"


def annotate_language_spans(text: str, primary_language: str | None = None) -> dict:
    """Return conservative transcript language spans with character offsets."""
    original = str(text or "")
    primary = _primary_language(primary_language)
    matches = list(_iter_word_spans(original))
    romanized_hindi_evidence = any(
        _script_for_token(original[start:end]) == "Latin"
        and (
            original[start:end].casefold().replace("’", "'")
            in STRONG_ROMANIZED_HINDI_WORDS
            or original[start:end].casefold().replace("’", "'")
            in ROMANIZED_HINDI_WORDS - AMBIGUOUS_ROMANIZED_HINDI_WORDS
        )
        for start, end in matches
    )
    tokens = []
    for start, end in matches:
        token = original[start:end]
        script = _script_for_token(token)
        language, confidence, method = _classify_token(
            token,
            script,
            primary,
            allow_primary_hint=not romanized_hindi_evidence,
        )
        tokens.append({
            "text": token,
            "char_start": start,
            "char_end": end,
            "script": script,
            "language": language,
            "confidence": confidence,
            "method": method,
        })

    spans = []
    for token in tokens:
        key = (token["language"], token["script"], token["method"])
        if spans and spans[-1]["_key"] == key:
            spans[-1]["char_end"] = token["char_end"]
            spans[-1]["text"] = original[spans[-1]["char_start"]:token["char_end"]]
            spans[-1]["tokens"].append(token["text"])
            spans[-1]["token_count"] += 1
            spans[-1]["confidence"] = min(spans[-1]["confidence"], token["confidence"])
        else:
            spans.append({
                "_key": key,
                "char_start": token["char_start"],
                "char_end": token["char_end"],
                "text": token["text"],
                "script": token["script"],
                "language": token["language"],
                "confidence": token["confidence"],
                "method": token["method"],
                "tokens": [token["text"]],
                "token_count": 1,
            })

    for span in spans:
        span.pop("_key", None)

    language_counts = Counter(
        token["language"] for token in tokens if token["language"] != "und"
    )
    languages_present = sorted(language_counts)
    script_counts = Counter(token["script"] for token in tokens)
    scripts_present = sorted(script_counts)
    code_switch = "hi" in language_counts and "en" in language_counts
    romanized_hindi = any(
        token["language"] == "hi" and token["script"] == "Latin"
        for token in tokens
    )
    primary_detected = (
        language_counts.most_common(1)[0][0] if language_counts else primary
    )

    return {
        "language_spans": spans,
        "language_evidence": dict(language_counts),
        "script_evidence": dict(script_counts),
        "languages_present": languages_present,
        "scripts_present": scripts_present,
        "dominant_language": primary_detected or "unknown",
        "code_switch_detected": code_switch,
        "mixed_script_detected": "Devanagari" in script_counts and "Latin" in script_counts,
        "romanized_hindi_detected": romanized_hindi,
        "span_unit": "transcript_character_offsets",
        "audio_timestamps_available": False,
    }
