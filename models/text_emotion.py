"""
SpeechMind text-emotion branch.

Use a locally fine-tuned MuRIL checkpoint for supported Indic/multilingual data.
When that checkpoint is absent, use a pretrained English-only emotion model for
English transcripts. Never apply the English model to Indic-language text.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_MODEL_NAME = "google/muril-base-cased"
ENGLISH_FALLBACK_MODEL = "j-hartmann/emotion-english-distilroberta-base"
CHECKPOINT_DIR = Path("checkpoints/text_emotion_muril")
EXPECTED_LABELS = {
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_label_order(metadata, model_config):
    """Resolve classifier-output order without assuming an arbitrary label order."""
    candidates = []
    if isinstance(metadata, dict):
        for key in ("class_order", "labels"):
            value = metadata.get(key)
            if isinstance(value, list) and value:
                candidates.append([str(label).lower().strip() for label in value])

    id2label = getattr(model_config, "id2label", None) or {}
    if id2label:
        ordered = [
            str(id2label.get(index, id2label.get(str(index), ""))).lower().strip()
            for index in range(len(id2label))
        ]
        if ordered and all(label and not label.startswith("label_") for label in ordered):
            candidates.append(ordered)

    for labels in candidates:
        if len(labels) == len(EXPECTED_LABELS) and set(labels) == EXPECTED_LABELS:
            return labels
    return None


class TextEmotionClassifier:
    def __init__(self):
        self.available = False
        self.model = None
        self.tokenizer = None
        self.labels = None
        self.model_source = None
        self.multilingual_checkpoint = False
        self.supported_languages = {"en"}
        self.load_error = None
        self.english_fallback_attempted = False

        if not CHECKPOINT_DIR.is_dir():
            self.load_error = f"Fine-tuned MuRIL checkpoint not found: {CHECKPOINT_DIR}"
            print(f"[TextEmotion] MuRIL unavailable: {self.load_error}")
            return

        try:
            metadata_path = CHECKPOINT_DIR / "speechmind_metadata.json"
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file() else {}
            )
            print(f"[TextEmotion] Loading fine-tuned MuRIL from {CHECKPOINT_DIR}...")
            tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))
            model = AutoModelForSequenceClassification.from_pretrained(
                str(CHECKPOINT_DIR)
            ).to(DEVICE)
            labels = _resolve_label_order(metadata, model.config)
            if labels is None:
                raise ValueError(
                    "Checkpoint labels are missing or incompatible. Expected exactly "
                    "these seven labels in the saved classifier order: "
                    + ", ".join(sorted(EXPECTED_LABELS))
                )
            if int(model.config.num_labels) != len(labels):
                raise ValueError(
                    f"Checkpoint has {model.config.num_labels} outputs but metadata "
                    f"defines {len(labels)} labels."
                )

            model.eval()
            self.tokenizer = tokenizer
            self.model = model
            self.labels = labels
            self.model_source = str(CHECKPOINT_DIR)
            raw_languages = metadata.get("languages", ["en"])
            if not isinstance(raw_languages, list) or not raw_languages:
                raw_languages = ["en"]
            self.supported_languages = {
                str(item).lower().strip() for item in raw_languages if str(item).strip()
            } or {"en"}
            self.multilingual_checkpoint = True
            self.available = True
            print(f"[TextEmotion] Fine-tuned MuRIL loaded with labels: {labels}")
        except Exception as error:
            self.load_error = str(error)
            print(f"[TextEmotion] MuRIL checkpoint unavailable: {error}")

    def _load_english_fallback(self):
        if self.english_fallback_attempted:
            return
        self.english_fallback_attempted = True
        try:
            print(
                "[TextEmotion] Loading English-only pretrained fallback "
                f"{ENGLISH_FALLBACK_MODEL}..."
            )
            tokenizer = AutoTokenizer.from_pretrained(ENGLISH_FALLBACK_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(
                ENGLISH_FALLBACK_MODEL
            ).to(DEVICE)
            labels = _resolve_label_order({}, model.config)
            if labels is None:
                raise ValueError(
                    "English fallback does not expose the expected seven emotion labels."
                )
            model.eval()
            self.tokenizer = tokenizer
            self.model = model
            self.labels = labels
            self.model_source = ENGLISH_FALLBACK_MODEL
            self.multilingual_checkpoint = False
            self.supported_languages = {"en"}
            self.available = True
            print(f"[TextEmotion] English fallback loaded with labels: {labels}")
        except Exception as error:
            self.load_error = (
                f"Could not load English fallback {ENGLISH_FALLBACK_MODEL}: {error}"
            )
            print(f"[TextEmotion] English fallback unavailable: {error}")

    @torch.no_grad()
    def predict(self, text: str, language: str = None) -> dict:
        language_code = str(language or "unknown").lower().strip()
        if not text or not text.strip():
            return {
                "emotion": "unavailable",
                "confidence": 0.0,
                "distribution": {},
                "available": False,
                "reason": "No transcript was produced.",
                "model_source": None,
            }

        if not self.available and language_code in ("en", "english"):
            self._load_english_fallback()

        if not self.available or self.model is None or self.tokenizer is None:
            reason = self.load_error or "No compatible trained checkpoint is available."
            if language_code not in ("en", "english"):
                reason = (
                    "No fine-tuned multilingual MuRIL checkpoint is installed. "
                    "The available fallback supports English transcripts only."
                )
            return {
                "emotion": "unavailable",
                "confidence": 0.0,
                "distribution": {},
                "available": False,
                "reason": reason,
                "model_source": None,
            }

        if language_code not in self.supported_languages and not (
            language_code == "english" and "en" in self.supported_languages
        ):
            return {
                "emotion": "unavailable",
                "confidence": 0.0,
                "distribution": {},
                "available": False,
                "reason": (
                    f"The loaded text-emotion checkpoint was not trained for {language_code}. "
                    "Supported language codes: " + ", ".join(sorted(self.supported_languages))
                ),
                "model_source": self.model_source,
            }

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(DEVICE)
        logits = self.model(**inputs).logits[0]
        probs = F.softmax(logits, dim=-1).float().cpu().numpy()
        if len(probs) != len(self.labels):
            raise ValueError(
                f"Text checkpoint has {len(probs)} outputs; expected {len(self.labels)} labels."
            )
        top_idx = int(probs.argmax())
        return {
            "emotion": self.labels[top_idx],
            "confidence": float(probs[top_idx]),
            "distribution": {
                label: float(probs[index])
                for index, label in enumerate(self.labels)
            },
            "available": True,
            "reason": None,
            "model_source": self.model_source,
        }
