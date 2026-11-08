"""
SpeechMind text-emotion branch.

A fine-tuned checkpoint is required. Base MuRIL with a newly initialized
classification head is not an emotion classifier, so this module deliberately
disables the branch when no trained checkpoint is present.
"""
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_MODEL_NAME = "google/muril-base-cased"
CHECKPOINT_DIR = "checkpoints/text_emotion_muril_bhaav"
# BHAAV Hindi text baseline: five source labels, including suspense.
# This is a story-text domain baseline, not a validated spoken-conversation model.
LABELS = ["anger", "joy", "suspense", "sadness", "neutral"]
MODEL_DOMAIN = "Hindi story text (BHAAV research baseline)"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TextEmotionClassifier:
    def __init__(self):
        self.available = False
        self.model = None
        self.tokenizer = None
        self.load_error = None

        if not os.path.isdir(CHECKPOINT_DIR):
            self.load_error = f"Fine-tuned checkpoint not found: {CHECKPOINT_DIR}"
            print(f"[TextEmotion] DISABLED: {self.load_error}")
            return

        try:
            print(f"[TextEmotion] Loading trained checkpoint from {CHECKPOINT_DIR}...")
            self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                CHECKPOINT_DIR
            ).to(DEVICE)
            self.model.eval()
            configured_labels = [
                self.model.config.id2label.get(index)
                or self.model.config.id2label.get(str(index))
                for index in range(len(LABELS))
            ]
            if configured_labels != LABELS:
                raise ValueError(
                    f"Checkpoint label order {configured_labels} does not match "
                    f"SpeechMind label order {LABELS}."
                )
            self.available = True
            print("[TextEmotion] Fine-tuned checkpoint loaded.")
        except Exception as error:
            self.load_error = str(error)
            self.model = None
            self.tokenizer = None
            print(f"[TextEmotion] DISABLED: checkpoint could not be loaded ({error}).")

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        if not self.available or self.model is None or self.tokenizer is None:
            return {
                "emotion": "unavailable",
                "confidence": 0.0,
                "distribution": {},
                "available": False,
                "reason": self.load_error or "No trained checkpoint is available.",
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
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        if len(probs) != len(LABELS):
            raise ValueError(
                f"Text checkpoint has {len(probs)} outputs; expected {len(LABELS)} labels."
            )
        top_idx = int(probs.argmax())
        return {
            "emotion": LABELS[top_idx],
            "confidence": float(probs[top_idx]),
            "distribution": {label: float(probs[i]) for i, label in enumerate(LABELS)},
            "available": True,
        }
