"""
SpeechMind voice-emotion inference.

Uses a fine-tuned local WavLM checkpoint only when it contains the metadata
written by the validated SpeechMind training workflow. Older experimental
checkpoints are ignored. If no approved-format checkpoint exists, the app
falls back to the configured published baseline.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForAudioClassification, AutoProcessor

BASE_MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"
FINETUNED_MODEL_DIR = Path("checkpoints/voice_emotion_wavlm")
SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_STEP_SECONDS = 6
EXPECTED_LABELS = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}


class VoiceEmotionRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metadata_path = FINETUNED_MODEL_DIR / "speechmind_metadata.json"
        config_path = FINETUNED_MODEL_DIR / "config.json"
        self.model_metadata = None
        if metadata_path.is_file() and config_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if set(metadata.get("labels", [])) == EXPECTED_LABELS:
                    self.model_source = str(FINETUNED_MODEL_DIR)
                    self.model_metadata = metadata
                else:
                    self.model_source = BASE_MODEL_NAME
            except (OSError, ValueError, TypeError):
                self.model_source = BASE_MODEL_NAME
        else:
            self.model_source = BASE_MODEL_NAME

        print(f"[VoiceEmotion] Loading {self.model_source} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_source)
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_source
        ).to(self.device)
        self.model.eval()
        print("[VoiceEmotion] Model loaded. Labels:", self.model.config.id2label)

    @staticmethod
    def _to_chunks(audio):
        window = WINDOW_SECONDS * SAMPLE_RATE
        step = WINDOW_STEP_SECONDS * SAMPLE_RATE
        if len(audio) <= window:
            return [audio]
        chunks = [audio[start:start + window] for start in range(0, len(audio), step)]
        if len(chunks) > 1 and len(chunks[-1]) < SAMPLE_RATE:
            chunks.pop()
        return chunks or [audio]

    @torch.inference_mode()
    def predict(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        waveform = np.asarray(audio, dtype=np.float32).squeeze()
        if waveform.ndim != 1:
            raise ValueError(
                f"Voice emotion model expects mono 1-D audio; received shape {waveform.shape}."
            )
        if waveform.size == 0:
            raise ValueError("Voice emotion model received empty audio.")
        if not np.isfinite(waveform).all():
            raise ValueError("Voice emotion model received NaN or infinite audio samples.")

        peak = float(np.max(np.abs(waveform)))
        if peak > 1.0:
            waveform = waveform / peak

        chunks = self._to_chunks(waveform)
        weighted_probabilities = None
        total_weight = 0.0
        for chunk in chunks:
            inputs = self.processor(
                chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            logits = self.model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1).float().cpu().numpy()
            weight = float(len(chunk))
            if weighted_probabilities is None:
                weighted_probabilities = probabilities * weight
            else:
                weighted_probabilities += probabilities * weight
            total_weight += weight

        probabilities = weighted_probabilities / max(total_weight, 1.0)
        id2label = self.model.config.id2label
        labels = [
            str(id2label.get(index, id2label.get(str(index), f"label_{index}"))).lower()
            for index in range(len(probabilities))
        ]
        distribution = {label: float(value) for label, value in zip(labels, probabilities)}
        top_index = int(np.argmax(probabilities))
        return {
            "emotion": labels[top_index],
            "confidence": round(float(probabilities[top_index]), 3),
            "distribution": distribution,
            "arousal": self._estimate_arousal(distribution),
            "inference_device": str(self.device),
            "inference_windows": len(chunks),
            "model_source": self.model_source,
            "validation_macro_f1": (
                self.model_metadata.get("best_validation_macro_f1")
                if self.model_metadata else None
            ),
        }

    @staticmethod
    def _estimate_arousal(distribution):
        """Qualitative prototype estimate; not a physiological measurement."""
        high = sum(
            distribution.get(label, 0.0)
            for label in ("angry", "anger", "fear", "fearful", "happy", "joy", "surprise")
        )
        low = sum(
            distribution.get(label, 0.0)
            for label in ("sad", "sadness", "neutral", "calm")
        )
        if high >= 0.60:
            return "High"
        if low >= 0.60:
            return "Low"
        return "Moderate"
