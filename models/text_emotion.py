"""
SpeechMind text-emotion analysis.

Preference order:
1. CPU-friendly HinEmo-9 char/word TF-IDF classifier for English, Hindi and Hinglish.
2. A compatible fine-tuned MuRIL checkpoint for additional languages.
3. English-only DistilRoBERTa fallback only when the transcript is confidently English.

The HinEmo-9 model has nine text categories. The voice branch remains a
separate seven-category acoustic model; fusion preserves the nine text labels.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_MODEL_NAME = "google/muril-base-cased"
ENGLISH_FALLBACK_MODEL = "j-hartmann/emotion-english-distilroberta-base"
HINEMO9_DIR = Path("checkpoints/text_emotion_hinemo9")
MURIL_DIR = Path("checkpoints/text_emotion_muril")

EXPECTED_EMOTIONS = {
    "love", "joy", "anger", "sadness", "fear",
    "surprise", "nostalgia", "devotion", "neutral",
}
SEVEN_EMOTIONS = {
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _softmax(values):
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    total = float(exp_scores.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Text emotion classifier returned invalid decision scores.")
    return exp_scores / total


class TextEmotionClassifier:
    def __init__(self):
        self.available = False
        self.model = None
        self.tokenizer = None
        self.labels = None
        self.model_source = None
        self.model_type = None
        self.svm_bundle = None
        self.multilingual_checkpoint = False
        self.supported_languages = set()
        self.load_error = None
        self.english_fallback_attempted = False

        if self._load_hinemo9():
            return
        if self._load_muril():
            return

        self.load_error = (
            "No trained Hinglish/Hindi text-emotion checkpoint was found. "
            f"Expected {HINEMO9_DIR} or {MURIL_DIR}."
        )
        print(f"[TextEmotion] Text branch awaiting training: {self.load_error}")

    def _load_hinemo9(self) -> bool:
        metadata_path = HINEMO9_DIR / "speechmind_metadata.json"
        head_path = HINEMO9_DIR / "text_emotion_head.joblib"
        if not metadata_path.is_file() or not head_path.is_file():
            return False
        try:
            import joblib
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("model_type") != "hinemo9_char_word_tfidf_linear_svc":
                raise ValueError("HinEmo-9 checkpoint model_type is incompatible.")
            labels = [
                str(label).lower().strip()
                for label in metadata.get("class_order", [])
            ]
            if len(labels) != len(EXPECTED_EMOTIONS) or set(labels) != EXPECTED_EMOTIONS:
                raise ValueError("HinEmo-9 checkpoint does not contain all nine emotion labels.")
            bundle = joblib.load(head_path)
            required = {"char_vectorizer", "word_vectorizer", "classifier"}
            if not isinstance(bundle, dict) or not required.issubset(bundle):
                raise ValueError("HinEmo-9 classifier file is incomplete.")
            classes = [
                str(label).lower().strip()
                for label in bundle["classifier"].classes_
            ]
            if classes != labels:
                raise ValueError("HinEmo-9 classifier classes do not match checkpoint metadata.")
            self.svm_bundle = bundle
            self.labels = labels
            self.model_source = str(HINEMO9_DIR)
            self.model_type = "hinemo9_char_word_tfidf_linear_svc"
            self.supported_languages = {"en", "hi", "hinglish", "en-or-romanized-hinglish"}
            self.multilingual_checkpoint = True
            self.available = True
            print(f"[TextEmotion] HinEmo-9 classifier loaded with labels: {labels}")
            return True
        except Exception as error:
            self.load_error = f"HinEmo-9 checkpoint could not be loaded: {error}"
            print(f"[TextEmotion] {self.load_error}")
            return False

    def _load_muril(self) -> bool:
        if not MURIL_DIR.is_dir():
            return False
        try:
            metadata_path = MURIL_DIR / "speechmind_metadata.json"
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file() else {}
            )
            tokenizer = AutoTokenizer.from_pretrained(str(MURIL_DIR))
            model = AutoModelForSequenceClassification.from_pretrained(
                str(MURIL_DIR)
            ).to(DEVICE)
            labels = [
                str(value).lower().strip()
                for value in metadata.get("class_order", metadata.get("labels", []))
            ]
            if len(labels) != int(model.config.num_labels) or not labels:
                raise ValueError("MuRIL checkpoint has no valid saved class order.")
            model.eval()
            self.tokenizer = tokenizer
            self.model = model
            self.labels = labels
            self.model_source = str(MURIL_DIR)
            self.model_type = "muril_finetuned"
            self.supported_languages = {
                str(value).lower().strip()
                for value in metadata.get("languages", ["en"])
            }
            self.multilingual_checkpoint = True
            self.available = True
            print(f"[TextEmotion] Fine-tuned MuRIL loaded with labels: {labels}")
            return True
        except Exception as error:
            self.load_error = f"MuRIL checkpoint could not be loaded: {error}"
            print(f"[TextEmotion] {self.load_error}")
            return False

    def _load_english_fallback(self):
        if self.english_fallback_attempted:
            return
        self.english_fallback_attempted = True
        try:
            print(
                "[TextEmotion] Loading English-only fallback "
                f"{ENGLISH_FALLBACK_MODEL}..."
            )
            tokenizer = AutoTokenizer.from_pretrained(ENGLISH_FALLBACK_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(
                ENGLISH_FALLBACK_MODEL
            ).to(DEVICE)
            labels = [
                str(model.config.id2label.get(i, model.config.id2label.get(str(i), "")))
                .lower().strip()
                for i in range(int(model.config.num_labels))
            ]
            normalized = [
                "joy" if label in {"happy", "happiness"} else label
                for label in labels
            ]
            if len(normalized) != len(SEVEN_EMOTIONS) or set(normalized) != SEVEN_EMOTIONS:
                raise ValueError("English fallback labels are not the seven SpeechMind classes.")
            model.eval()
            self.tokenizer = tokenizer
            self.model = model
            self.labels = normalized
            self.model_source = ENGLISH_FALLBACK_MODEL
            self.model_type = "english_distilroberta"
            self.supported_languages = {"en"}
            self.multilingual_checkpoint = False
            self.available = True
        except Exception as error:
            self.load_error = f"English fallback unavailable: {error}"
            print(f"[TextEmotion] {self.load_error}")

    @torch.no_grad()
    def predict(self, text: str, language: str = None) -> dict:
        language_profile = str(language or "unknown").lower().strip()
        if not text or not text.strip():
            return {
                "emotion": "unavailable",
                "confidence": 0.0,
                "distribution": {},
                "available": False,
                "reason": "No transcript was produced.",
                "model_source": None,
            }

        if self.model_type == "hinemo9_char_word_tfidf_linear_svc":
            if language_profile not in self.supported_languages and (
                language_profile != "code_mixed_or_mixed_script"
            ):
                return {
                    "emotion": "unavailable",
                    "confidence": 0.0,
                    "distribution": {},
                    "available": False,
                    "reason": (
                        "The Hinglish text classifier is trained on English, Hindi "
                        "and romanized Hinglish. This transcript's language profile "
                        f"('{language_profile}') is outside that training scope."
                    ),
                    "model_source": self.model_source,
                }
            char_features = self.svm_bundle["char_vectorizer"].transform([text])
            word_features = self.svm_bundle["word_vectorizer"].transform([text])
            from scipy.sparse import hstack
            features = hstack([char_features, word_features], format="csr")
            scores = self.svm_bundle["classifier"].decision_function(features)[0]
            probabilities = _softmax(scores)
            distribution = {
                label: float(probabilities[index])
                for index, label in enumerate(self.labels)
            }
            emotion = max(distribution, key=distribution.get)
            return {
                "emotion": emotion,
                "confidence": distribution[emotion],
                "distribution": distribution,
                "available": True,
                "reason": None,
                "model_source": self.model_source,
                "score_type": "uncalibrated_linear_svc_decision_softmax",
            }

        # Do not silently apply an English-only model to romanized Hinglish
        # or to transcripts whose language remains uncertain.
        if not self.available and language_profile == "en":
            self._load_english_fallback()

        if not self.available or self.model is None or self.tokenizer is None:
            return {
                "emotion": "unavailable",
                "confidence": 0.0,
                "distribution": {},
                "available": False,
                "reason": self.load_error or "No compatible text-emotion model is installed.",
                "model_source": None,
            }

        if language_profile not in self.supported_languages:
            return {
                "emotion": "unavailable",
                "confidence": 0.0,
                "distribution": {},
                "available": False,
                "reason": (
                    f"The text-emotion checkpoint was not trained for "
                    f"'{language_profile}'. Supported: "
                    + ", ".join(sorted(self.supported_languages))
                ),
                "model_source": self.model_source,
            }

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(DEVICE)
        logits = self.model(**inputs).logits[0]
        probabilities = F.softmax(logits, dim=-1).float().cpu().numpy()
        if len(probabilities) != len(self.labels):
            raise ValueError(
                f"Text checkpoint has {len(probabilities)} outputs; "
                f"expected {len(self.labels)} labels."
            )
        distribution = {
            label: float(probabilities[index])
            for index, label in enumerate(self.labels)
        }
        emotion = max(distribution, key=distribution.get)
        return {
            "emotion": emotion,
            "confidence": distribution[emotion],
            "distribution": distribution,
            "available": True,
            "reason": None,
            "model_source": self.model_source,
            "score_type": "uncalibrated_softmax",
        }
