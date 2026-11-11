"""IndicTrans2 adapter with an OS-isolated translation runtime."""
from __future__ import annotations

import os

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
    def __init__(self, provider=None):
        if provider is None:
            if os.name == "nt":
                from models.wsl_translation_provider import WslIndicTransProvider
                provider = WslIndicTransProvider()
                self.provider_name = "wsl_linux_indictrans2"
            else:
                from models.srota_provider import SrotaASRProvider
                provider = SrotaASRProvider()
                self.provider_name = "isolated_indictrans2_worker"
        else:
            self.provider_name = (
                "injected_translation_provider"
                if provider.__class__.__name__ != "SrotaASRProvider"
                else "isolated_indictrans2_worker"
            )
        self.provider = provider
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
