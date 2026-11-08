"""
SpeechMind - Text emotion branch, powered by Google's MuRIL.

Uses a fine-tuned MuRIL checkpoint if one exists at CHECKPOINT_DIR (see the
training script you'll run to fine-tune on GoEmotions + BHAAV + Hinglish
emotion data). If no fine-tuned checkpoint is found yet, falls back to base
MuRIL with an UNTRAINED classification head so the pipeline still runs
end-to-end during development - predictions will look close to random until
you fine-tune, which is expected and not a bug.

Output labels are the same 7-class set fusion_layer.py's UNIFIED_EMOTIONS
uses directly (joy, sadness, anger, fear, surprise, disgust, neutral), so no
extra label mapping is needed on the text side.
"""
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

BASE_MODEL_NAME = "google/muril-base-cased"
CHECKPOINT_DIR = "checkpoints/text_emotion_muril"

LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TextEmotionClassifier:
    def __init__(self):
        if os.path.isdir(CHECKPOINT_DIR):
            print(f"[TextEmotion] Loading fine-tuned checkpoint from {CHECKPOINT_DIR}")
            self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(DEVICE)
        else:
            print(
                "[TextEmotion] WARNING: no fine-tuned checkpoint found at "
                f"'{CHECKPOINT_DIR}'. Loading base MuRIL with an UNTRAINED "
                "classification head - fine-tune before trusting these predictions."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            config = AutoConfig.from_pretrained(BASE_MODEL_NAME, num_labels=len(LABELS))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL_NAME, config=config
            ).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """
        Returns:
            {
                "emotion": "joy",
                "confidence": 0.82,
                "distribution": {"joy": 0.82, "sadness": 0.03, ...}
            }
        """
        if not text or not text.strip():
            # Empty transcript (e.g. silence or failed ASR) - return a flat,
            # low-confidence neutral prediction instead of crashing.
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "distribution": {label: 1.0 / len(LABELS) for label in LABELS},
            }

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        logits = self.model(**inputs).logits[0]
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        top_idx = int(probs.argmax())
        distribution = {label: float(probs[i]) for i, label in enumerate(LABELS)}

        return {
            "emotion": LABELS[top_idx],
            "confidence": float(probs[top_idx]),
            "distribution": distribution,
        }
