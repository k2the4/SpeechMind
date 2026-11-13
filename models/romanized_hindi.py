"""Optional local Romanized-Hindi to Devanagari transliteration."""
import re
import threading

_ENGINE = None
_ENGINE_LOCK = threading.Lock()
_CALL_LOCK = threading.Lock()


def _get_engine():
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            try:
                from ai4bharat.transliteration import XlitEngine
            except ImportError as error:
                raise RuntimeError(
                    "Install requirements_hinglish_transliteration.txt in the SpeechMind "
                    "environment to enable optional local IndicXlit. No hosted API is used."
                ) from error
            _ENGINE = XlitEngine("hi", beam_width=4, rescore=True)
        return _ENGINE


def transliterate_romanized_hindi(text, language_spans, engine=None):
    original = str(text or "")
    spans = [
        span for span in (language_spans or [])
        if isinstance(span, dict) and span.get("language") == "hi"
        and span.get("script") == "Latin"
        and isinstance(span.get("char_start"), int)
        and isinstance(span.get("char_end"), int)
    ]
    if not spans:
        return {"available": False, "text": original, "replacements": 0,
                "model": None, "error": "No confidently tagged Romanized-Hindi words were found."}
    try:
        active = engine if engine is not None else _get_engine()
        replacements = []
        lock = _CALL_LOCK if engine is None else _NoLock()
        with lock:
            for span in spans:
                start, end = span["char_start"], span["char_end"]
                if start < 0 or end > len(original) or start >= end:
                    continue
                segment = original[start:end]
                for match in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)*", segment):
                    word = match.group(0)
                    value = active.translit_word(word, lang_code="hi", topk=1)
                    if isinstance(value, dict):
                        value = value.get("hi", [])
                    candidate = value.strip() if isinstance(value, str) else (
                        str(value[0]).strip() if isinstance(value, (list, tuple)) and value else ""
                    )
                    if candidate and candidate.casefold() != word.casefold():
                        replacements.append((start + match.start(), start + match.end(), candidate))
        converted = original
        for start, end, candidate in reversed(replacements):
            converted = converted[:start] + candidate + converted[end:]
        if not replacements:
            return {"available": False, "text": original, "replacements": 0,
                    "model": "AI4Bharat IndicXlit", "error": "IndicXlit returned no usable Hindi transliterations."}
        return {"available": True, "text": converted, "replacements": len(replacements),
                "model": "AI4Bharat IndicXlit", "error": None}
    except Exception as error:
        return {"available": False, "text": original, "replacements": 0,
                "model": "AI4Bharat IndicXlit", "error": f"{type(error).__name__}: {error}"}


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False
