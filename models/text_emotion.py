"""
SpeechMind text-emotion branch.

A fine-tuned multilingual MuRIL checkpoint handles Hindi and Hinglish. The
model can be single-label (legacy checkpoints) or multi-label (six emotion
signals with neutral derived from no active signal). English uses an independent
English-only fallback and never replaces the multilingual checkpoint.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models.text_emotion_format import EMOTION_LABELS, format_multilabel_scores

BASE_MODEL_NAME = "google/muril-base-cased"
ENGLISH_FALLBACK_MODEL = "j-hartmann/emotion-english-distilroberta-base"
CHECKPOINT_DIR = Path("checkpoints/text_emotion_muril")
EXPECTED_LABELS = {
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"
}
EXPECTED_MULTILABEL_LABELS = set(EMOTION_LABELS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _metadata_label_candidates(metadata, model_config):
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
    return candidates


def _resolve_label_order(metadata, model_config):
    """Resolve legacy seven-class single-label output order."""
    for labels in _metadata_label_candidates(metadata, model_config):
        if len(labels) == len(EXPECTED_LABELS) and set(labels) == EXPECTED_LABELS:
            return labels
    return None


def _resolve_multilabel_order(metadata, model_config):
    """Resolve the six independent non-neutral model outputs."""
    for labels in _metadata_label_candidates(metadata, model_config):
        if (
            len(labels) == len(EXPECTED_MULTILABEL_LABELS)
            and set(labels) == EXPECTED_MULTILABEL_LABELS
        ):
            return labels
    return None


class TextEmotionClassifier:
    def __init__(self):
        self.available = False
        self.model = None
        self.tokenizer = None
        self.labels = None
        self.model_source = None
        self.task_type = "single_label"
        self.multilingual_checkpoint = False
        self.supported_languages = set()
        self.load_error = None
        self.english_fallback_attempted = False
        self.english_model = None
        self.english_tokenizer = None
        self.english_labels = None
        self.english_load_error = None
        self.thresholds = {}
        self.max_length = 128

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

            task_type = str(metadata.get("task_type") or "single_label")
            if task_type == "multilabel_six_plus_neutral":
                labels = _resolve_multilabel_order(metadata, model.config)
                if labels is None:
                    raise ValueError(
                        "Multi-label checkpoint metadata must define exactly these "
                        "six outputs: " + ", ".join(EMOTION_LABELS)
                    )
            else:
                labels = _resolve_label_order(metadata, model.config)
                if labels is None:
                    raise ValueError(
                        "Checkpoint labels are missing or incompatible. Expected "
                        "the seven SpeechMind labels: "
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
            self.task_type = task_type
            self.thresholds = metadata.get("thresholds") or {}
            self.max_length = int(metadata.get("max_length", 128))
            if self.max_length < 8 or self.max_length > 512:
                raise ValueError(f"Invalid checkpoint max_length: {self.max_length}")
            raw_languages = metadata.get("languages", ["en"])
            if not isinstance(raw_languages, list) or not raw_languages:
                raw_languages = ["en"]
            self.supported_languages = {
                str(item).lower().strip() for item in raw_languages if str(item).strip()
            }
            self.multilingual_checkpoint = True
            self.available = True
            print(
                f"[TextEmotion] Fine-tuned MuRIL loaded; task={task_type}; "
                f"labels={labels}; languages={sorted(self.supported_languages)}"
            )
        except Exception as error:
            self.load_error = str(error)
            print(f"[TextEmotion] MuRIL checkpoint unavailable: {error}")

    def _load_english_fallback(self):
        if self.english_fallback_attempted:
            return self.english_model is not None
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
            self.english_tokenizer = tokenizer
            self.english_model = model
            self.english_labels = labels
            self.supported_languages.add("en")
            print(f"[TextEmotion] English fallback loaded with labels: {labels}")
            return True
        except Exception as error:
            self.english_load_error = (
                f"Could not load English fallback {ENGLISH_FALLBACK_MODEL}: {error}"
            )
            print(f"[TextEmotion] English fallback unavailable: {error}")
            return False

    @staticmethod
    def _unavailable(reason: str, model_source=None) -> dict:
        return {
            "emotion": "unavailable",
            "confidence": 0.0,
            "distribution": {},
            "secondary_emotions": [],
            "available": False,
            "reason": reason,
            "model_source": model_source,
        }

    @torch.inference_mode()
    def predict(self, text: str, language: str = None) -> dict:
        language_code = str(language or "unknown").lower().strip()
        if not text or not text.strip():
            return self._unavailable("No transcript was produced.")

        if language_code in ("en", "english"):
            if not self._load_english_fallback():
                return self._unavailable(
                    self.english_load_error or "English emotion model is unavailable."
                )
            model = self.english_model
            tokenizer = self.english_tokenizer
            labels = self.english_labels
            model_source = ENGLISH_FALLBACK_MODEL
            task_type = "single_label"
        else:
            if not self.available or self.model is None or self.tokenizer is None:
                return self._unavailable(
                    "No fine-tuned multilingual MuRIL checkpoint is installed. "
                    "The available fallback supports English transcripts only."
                )
            if language_code not in self.supported_languages:
                return self._unavailable(
                    f"The loaded text-emotion checkpoint was not trained for {language_code}. "
                    "Supported language codes: " + ", ".join(sorted(self.supported_languages)),
                    self.model_source,
                )
            model = self.model
            tokenizer = self.tokenizer
            labels = self.labels
            model_source = self.model_source
            task_type = self.task_type

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(DEVICE)
        logits = model(**inputs).logits[0]

        if task_type == "multilabel_six_plus_neutral":
            probabilities = torch.sigmoid(logits).float().cpu().numpy()
            scores = {
                label: float(probabilities[index])
                for index, label in enumerate(labels)
            }
            formatted = format_multilabel_scores(scores, self.thresholds)
            decision_thresholds = {
                label: float(self.thresholds.get(label, 0.5))
                for label in labels
            }
            active_emotions = [
                label for label, score in scores.items()
                if score >= decision_thresholds[label]
            ]
            return {
                "emotion": formatted["emotion"],
                "confidence": float(formatted["confidence"]),
                "distribution": formatted["distribution"],
                "secondary_emotions": formatted["secondary_emotions"],
                "multi_label_scores": formatted["multi_label_scores"],
                "decision_thresholds": decision_thresholds,
                "active_emotions": active_emotions,
                "available": True,
                "reason": None,
                "model_source": model_source,
                "task_type": task_type,
            }

        probs = F.softmax(logits, dim=-1).float().cpu().numpy()
        if len(probs) != len(labels):
            raise ValueError(
                f"Text checkpoint has {len(probs)} outputs; expected {len(labels)} labels."
            )
        top_idx = int(probs.argmax())
        return {
            "emotion": labels[top_idx],
            "confidence": float(probs[top_idx]),
            "distribution": {
                label: float(probs[index])
                for index, label in enumerate(labels)
            },
            "secondary_emotions": [],
            "available": True,
            "reason": None,
            "model_source": model_source,
            "task_type": task_type,
        }
