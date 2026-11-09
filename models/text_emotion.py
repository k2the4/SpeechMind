"""
SpeechMind text-emotion branch.

A fine-tuned checkpoint is required. Base MuRIL with a newly initialized
classification head is not an emotion classifier, so this module deliberately
disables the branch when no trained, label-compatible checkpoint is present.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_MODEL_NAME = "google/muril-base-cased"
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
        self.load_error = None

        metadata_path = CHECKPOINT_DIR / "speechmind_metadata.json"
        if not CHECKPOINT_DIR.is_dir():
            self.load_error = f"Fine-tuned checkpoint not found: {CHECKPOINT_DIR}"
            print(f"[TextEmotion] DISABLED: {self.load_error}")
            return

        try:
            metadata = {}
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            print(f"[TextEmotion] Loading trained checkpoint from {CHECKPOINT_DIR}...")
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
            self.available = True
            print(f"[TextEmotion] Fine-tuned checkpoint loaded with labels: {labels}")
        except Exception as error:
            self.load_error = str(error)
            self.model = None
            self.tokenizer = None
            self.labels = None
            print(f"[TextEmotion] DISABLED: checkpoint could not be loaded ({error}).")

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        if not self.available or self.model is None or self.tokenizer is None:
            return {
                "emotion": "unavailable",
                "confidence": 0.0,
                "distribution": {},
                "available": False,
                "reason": self.load_error or "No compatible trained checkpoint is available.",
            }

        if not text or not text.strip():
            return {
                "emotion": "unavailable",
                "confidence": 0.0,
                "distribution": {},
                "available": False,
                "reason": "No transcript was produced.",
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
        }
