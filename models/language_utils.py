"""Lightweight language/script helpers. These are hints, never ground truth."""
import re
import unicodedata

_TOKEN_RE = re.compile(r"[\w']+", flags=re.UNICODE)

# Deliberately distinctive Romanized Hindi tokens; this catches common
# code-mixed phrases without treating ordinary English function words as Hindi.
_ROMANIZED_HINDI = {
    "yaar", "mujhe", "mujhko", "maine", "mera", "meri", "mere",
    "tumhara", "tumhari", "tumhe", "aapka", "aapki", "apna", "apni",
    "nahi", "nahin", "kyun", "kyunki", "kya", "kaise", "kabhi",
    "bahut", "bohot", "accha", "achha", "acha", "theek", "thik",
    "hain", "hoon", "houn", "tha", "thi", "the", "raha", "rahi",
    "rahe", "gaya", "gayi", "gaye", "hogaya", "hogayi", "hoga",
    "chahiye", "karna", "karke", "karo", "karo", "matlab",
    "kal", "aaj", "abhi", "phir", "lekin", "magar", "aur",
    "mein", "mujhse", "tumse", "isse", "usse", "kaafi",
}


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= char <= "\u097f" for char in str(text or ""))


def has_latin(text: str) -> bool:
    return any(("a" <= char.lower() <= "z") for char in str(text or ""))


def is_code_mixed_script(text: str) -> bool:
    text = str(text or "")
    return has_devanagari(text) and has_latin(text)


def looks_romanized_hindi(text: str, minimum_distinctive_tokens: int = 1) -> bool:
    tokens = {token.casefold() for token in _TOKEN_RE.findall(str(text or ""))}
    return len(tokens & _ROMANIZED_HINDI) >= minimum_distinctive_tokens


def normalize_language_code(language) -> str:
    if not language:
        return "unknown"
    value = unicodedata.normalize("NFKC", str(language)).lower().strip()
    aliases = {"english": "en", "hindi": "hi"}
    return aliases.get(value, value)


def infer_transcript_language(text: str, provider_language=None, provider_languages=None) -> str:
    """Prefer explicit mixed/script evidence; never infer a token language map."""
    text = str(text or "")
    languages = {
        normalize_language_code(item)
        for item in (provider_languages or [])
        if normalize_language_code(item) not in {"", "unknown"}
    }
    if len(languages) > 1:
        return "mixed"
    if is_code_mixed_script(text) or looks_romanized_hindi(text):
        return "mixed"
    language = normalize_language_code(provider_language)
    return language if language != "unknown" else "unknown"
