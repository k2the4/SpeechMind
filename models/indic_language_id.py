"""Optional IndicLID post-transcription language identification.

IndicLID identifies language/script from text, including Romanized Indian
languages. It is deliberately downstream of ASR: speakers never choose a
language, and ASR never routes an entire recording based on one language guess.
The fastText native and roman models are downloaded separately by the setup BAT.
"""
from __future__ import annotations

from pathlib import Path
import re

from models.bodhan_client import BodhanClient

CHECKPOINT_ROOT = Path("checkpoints/indiclid")
NATIVE_MODEL_PATH = CHECKPOINT_ROOT / "ftn" / "model_baseline_roman.bin"
ROMAN_MODEL_PATH = CHECKPOINT_ROOT / "ftr" / "model_baseline_roman.bin"

LANGUAGE_NAMES = {
    "asm": "as", "ben": "bn", "brx": "brx", "doi": "doi",
    "eng": "en", "guj": "gu", "hin": "hi", "kan": "kn",
    "kas": "ks", "kok": "kok", "mai": "mai", "mal": "ml",
    "mar": "mr", "mni": "mni", "nep": "ne", "ori": "or",
    "pan": "pa", "san": "sa", "sat": "sat", "snd": "sd",
    "tam": "ta", "tel": "te", "urd": "ur",
}


class IndicLanguageIdentifier:
    def __init__(self):
        self.native_model = None
        self.roman_model = None
        self.available = False
        self.load_error = None

        if not NATIVE_MODEL_PATH.is_file() or not ROMAN_MODEL_PATH.is_file():
            self.load_error = (
                "IndicLID weights not installed. Run DOWNLOAD_INDICLID_MODELS.bat "
                "to enable native and romanized Indian-language identification."
            )
            print(f"[IndicLID] Script-based fallback active: {self.load_error}")
            return

        try:
            import fasttext
            self.native_model = fasttext.load_model(str(NATIVE_MODEL_PATH))
            self.roman_model = fasttext.load_model(str(ROMAN_MODEL_PATH))
            self.available = True
            print("[IndicLID] Native-script and romanized fastText models loaded.")
        except Exception as error:
            self.load_error = str(error)
            print(f"[IndicLID] Could not load models; using script heuristic: {error}")

    @staticmethod
    def _is_roman_script(text: str) -> bool:
        chars = [
            char for char in text
            if not char.isspace() and not re.match(r"[^\w]", char)
        ]
        if not chars:
            return True
        latin = sum(1 for char in chars if char.isascii() and char.isalnum())
        return (latin / len(chars)) > 0.5

    @staticmethod
    def _windows(text: str, window_words: int = 4, stride: int = 3) -> list[str]:
        words = text.split()
        if len(words) <= window_words:
            return [text]
        windows = []
        for start in range(0, len(words), stride):
            part = words[start:start + window_words]
            if part:
                windows.append(" ".join(part))
            if start + window_words >= len(words):
                break
        return windows or [text]

    @staticmethod
    def summarize_labels(labels: list[str]) -> str:
        """Convert IndicLID segment labels into a product-level language profile."""
        normalized = [
            str(label).replace("__label__", "").strip()
            for label in labels
        ]
        languages = {
            LANGUAGE_NAMES[label.split("_", 1)[0]]
            for label in normalized
            if label.split("_", 1)[0] in LANGUAGE_NAMES
        }
        if not languages:
            return "undetermined"

        has_english = "eng_Latn" in normalized
        has_hindi_roman = "hin_Latn" in normalized
        has_hindi_native = "hin_Deva" in normalized

        # Romanized Hindi is the core Hinglish use case even when a short clip
        # contains no English segment. Hindi + English segments are also Hinglish.
        if has_hindi_roman or (has_english and has_hindi_native):
            return "hinglish"
        if languages == {"en"}:
            return "en"
        if languages == {"hi"}:
            return "hi"
        if languages == {"mr"}:
            return "mr"
        if languages == {"bn"}:
            return "bn"
        if languages == {"ta"}:
            return "ta"
        if languages == {"te"}:
            return "te"
        if languages == {"pa"}:
            return "pa"
        if len(languages) == 1:
            return next(iter(languages))
        return "code_mixed"

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                **BodhanClient.language_profile(""),
                "language_identifier": "script_heuristic",
                "language_segments": [],
                "language_id_available": False,
            }

        if not self.available:
            fallback = BodhanClient.language_profile(text)
            profile = fallback["language_profile"]
            # Preserve clear native-script identifications from script ranges.
            script_to_profile = {
                "hi_or_mr": "hi_or_mr",
                "bn": "bn",
                "pa": "pa",
                "ta": "ta",
                "te": "te",
            }
            if fallback["script_evidence"] == ["latin"]:
                profile = "english_or_romanized_hinglish"
            elif len(fallback["script_evidence"]) == 1:
                profile = script_to_profile.get(
                    fallback["script_evidence"][0], profile
                )
            return {
                **fallback,
                "language_profile": profile,
                "language_identifier": "script_heuristic",
                "language_segments": [],
                "language_id_available": False,
            }

        labels = []
        segments = []
        for window in self._windows(text):
            model = self.roman_model if self._is_roman_script(window) else self.native_model
            try:
                predicted_labels, predicted_scores = model.predict(
                    window.replace("\n", " ").strip(), k=1
                )
                label = str(predicted_labels[0][0]).replace("__label__", "")
                score = float(predicted_scores[0][0])
            except Exception as error:
                segments.append({
                    "text": window,
                    "label": "other",
                    "score": None,
                    "error": str(error),
                })
                continue
            labels.append(label)
            segments.append({
                "text": window,
                "label": label,
                "score": round(score, 4),
                "script_model": "roman" if self._is_roman_script(window) else "native",
            })

        profile = self.summarize_labels(labels)
        return {
            "language_profile": profile,
            "language_profile_note": (
                "IndicLID is a text-level language estimate. Romanized Hindi "
                "identification is less reliable than native-script identification."
            ),
            "script_evidence": sorted({
                str(segment.get("label", "other")).split("_")[-1]
                for segment in segments
                if segment.get("label") and segment.get("label") != "other"
            }),
            "language_identifier": "ai4bharat_indiclid_fasttext",
            "language_segments": segments,
            "language_id_available": bool(labels),
        }
