"""Thin adapter for IndicTrans2 running in the isolated model worker.

This module intentionally does not import Transformers or Torch in the main
SpeechMind environment. IndicTransToolkit and IndicTrans2 stay inside
.venv_srota, alongside the optional Srota worker.
"""
from __future__ import annotations

from models.srota_provider import SrotaASRProvider

MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

# IndicTrans2 uses FLORES-style language codes rather than plain ISO codes.
FLORES_CODE = {
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ml": "mal_Mlym",
    "kn": "kan_Knda",
    "ur": "urd_Arab",
}


class Translator:
    def __init__(self, provider: SrotaASRProvider | None = None):
        self.provider = provider or SrotaASRProvider()
        self.loaded = False
        self.model_name = MODEL_NAME
        self.last_inference_seconds = None

    def to_english(self, text: str, src_lang: str) -> str:
        """Translate original Hindi/Indic text to English without mutating it."""
        if not text or not text.strip():
            return text
        language = str(src_lang or "").lower().strip()
        if language in ("en", "english"):
            return text
        src_flores = FLORES_CODE.get(language)
        if src_flores is None:
            raise ValueError(
                f"IndicTrans2 translation is not configured for language: {language or 'unknown'}"
            )

        result = self.provider.translate_to_english(text, src_flores)
        translated = str(result.get("translated_text") or "").strip()
        if not translated:
            raise RuntimeError("IndicTrans2 returned an empty translation.")
        self.loaded = True
        self.last_inference_seconds = result.get("inference_seconds")
        return translated
