"""
SpeechMind English-meaning translation.

Bodhan Indic-Translate is the preferred code-mixed path because its API infers
the source language and explicitly supports romanized/code-mixed input. It is
only used when explicitly configured with a local API key. The existing local
IndicTrans2 distilled model remains a fallback for supported Indic-script
input; it is not treated as a general Hinglish translator.
"""
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"
BODHAN_TRANSLATE_ENDPOINT = "https://api.bodhan.ai/translate"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= char <= "\u097f" for char in text)


def _has_latin(text: str) -> bool:
    return any("a" <= char.lower() <= "z" for char in text)


def is_code_mixed_script(text: str) -> bool:
    return _has_devanagari(text) and _has_latin(text)


class Translator:
    def __init__(self):
        self.mode = os.getenv("SPEECHMIND_TRANSLATION_BACKEND", "auto").strip().lower()
        if self.mode not in {"auto", "bodhan", "indictrans2", "none"}:
            raise ValueError(
                "SPEECHMIND_TRANSLATION_BACKEND must be auto, bodhan, indictrans2, or none."
            )
        self.api_key = (
            os.getenv("BODHAN_TRANSLATE_API_KEY")
            or os.getenv("BODHAN_API_KEY")
            or ""
        ).strip()
        self.tokenizer = None
        self.model = None
        self.ip = None
        self.backend = None
        self.last_reason = None

    def _translate_with_bodhan(self, text: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "Bodhan translation was selected but BODHAN_TRANSLATE_API_KEY is missing."
            )
        try:
            import requests
        except ImportError as error:
            raise RuntimeError("Bodhan translation requires the requests package.") from error

        response = requests.post(
            BODHAN_TRANSLATE_ENDPOINT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"text": text, "target_language": "en"},
            timeout=(15, 120),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(f"Bodhan translation error: {payload['error']}")
        if payload.get("truncated"):
            raise RuntimeError("Bodhan translation was truncated.")
        translated = payload.get("translation")
        if not isinstance(translated, str) or not translated.strip():
            raise RuntimeError("Bodhan translation returned no translation text.")
        self.backend = "bodhan-indic-translate"
        return translated.strip()

    def _load_indictrans2(self):
        if self.model is not None:
            return
        from IndicTransToolkit.processor import IndicProcessor

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        ).to(DEVICE)
        self.model.eval()
        self.ip = IndicProcessor(inference=True)

    @torch.no_grad()
    def _translate_with_indictrans2(self, text: str, src_lang: str) -> str:
        src_flores = FLORES_CODE.get(src_lang)
        if src_flores is None:
            raise RuntimeError(
                f"IndicTrans2 does not have a configured source mapping for '{src_lang}'."
            )
        self._load_indictrans2()
        batch = self.ip.preprocess_batch(
            [text], src_lang=src_flores, tgt_lang="eng_Latn"
        )
        inputs = self.tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        generated = self.model.generate(
            **inputs, max_length=256, num_beams=5
        )
        decoded = self.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )
        translated = self.ip.postprocess_batch(decoded, lang="eng_Latn")[0]
        if not translated.strip():
            raise RuntimeError("IndicTrans2 returned an empty translation.")
        self.backend = "indictrans2-distilled"
        return translated.strip()

    def to_english(self, text: str, src_lang: str) -> str | None:
        if not text or not text.strip():
            self.last_reason = "No transcript was produced."
            return None

        language = str(src_lang or "unknown").lower().strip()
        mixed_script = is_code_mixed_script(text)
        hosted_requested = self.mode == "bodhan" or (
            self.mode == "auto" and bool(self.api_key)
        )

        if self.mode == "none":
            self.backend = "none"
            self.last_reason = "Translation is disabled in configuration."
            return None

        if hosted_requested:
            try:
                result = self._translate_with_bodhan(text)
                self.last_reason = None
                return result
            except Exception as error:
                self.last_reason = str(error)
                if self.mode == "bodhan":
                    return None
                # Auto mode can fall back only for clearly supported,
                # Indic-script monolingual input.
                if mixed_script:
                    return None

        if language in {"en", "english"} and not mixed_script:
            # For Latin-only text, Romanized Hinglish can be indistinguishable
            # from English from script alone. Do not call it a translation.
            self.backend = "unchanged_or_romanized_unknown"
            self.last_reason = (
                "The transcript is Latin-script and may be English or Romanized "
                "Hinglish. Configure Bodhan Indic-Translate to resolve code-mixed meaning."
            )
            return None

        if mixed_script or (language in FLORES_CODE and not _has_devanagari(text)):
            self.last_reason = (
                "The local IndicTrans2 fallback is not treated as a reliable "
                "romanized/code-mixed translator. Configure Bodhan Indic-Translate."
            )
            return None

        if language not in FLORES_CODE or not _has_devanagari(text):
            self.last_reason = (
                "No compatible local translation route is available for this transcript."
            )
            return None

        if self.mode == "bodhan":
            self.last_reason = "Bodhan translation is configured but unavailable."
            return None

        try:
            result = self._translate_with_indictrans2(text, language)
            self.last_reason = None
            return result
        except Exception as error:
            self.last_reason = str(error)
            return None
