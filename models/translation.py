"""
SpeechMind - Translation branch.

Converts the original-language transcript into English for the final
"Meaning (English)" field shown in app.py. This runs AFTER meaning/emotion
have already been extracted from the original-language text/audio - never
before - so nuance isn't lost prior to analysis.

Uses AI4Bharat's distilled IndicTrans2 (200M) rather than the full 1B model,
since the distilled version is realistic to run on a free Colab GPU or even
CPU, which matters for a student project.
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# IndicTrans2 uses FLORES-style language codes rather than plain ISO codes.
FLORES_CODE = {
    "hi": "hin_Deva",
    "en": "eng_Latn",
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
    def __init__(self):
        from IndicTransToolkit.processor import IndicProcessor  # pip install IndicTransToolkit

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
        self.model.eval()
        self.ip = IndicProcessor(inference=True)

    @torch.no_grad()
    def to_english(self, text: str, src_lang: str) -> str:
        """
        src_lang: two-letter code (e.g. "hi") as returned by models/asr.py.
        If src_lang is already English (or unrecognized), returns text unchanged
        rather than crashing the pipeline.
        """
        if not text or not text.strip():
            return text

        if src_lang == "en":
            return text

        src_flores = FLORES_CODE.get(src_lang)
        if src_flores is None:
            # Unknown/unsupported language tag - don't fail the whole pipeline,
            # just skip translation and show the original transcript.
            return text

        try:
            batch = self.ip.preprocess_batch([text], src_lang=src_flores, tgt_lang="eng_Latn")
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            generated = self.model.generate(**inputs, max_length=256, num_beams=5)
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            translated = self.ip.postprocess_batch(decoded, lang="eng_Latn")
            return translated[0]
        except Exception as e:
            print(f"[Translation] Failed ({e}); falling back to original transcript")
            return text
