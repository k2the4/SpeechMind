"""
SpeechMind voice-emotion inference.

Prefers the CPU-friendly, locally trained frozen-WavLM/SVM checkpoint when
available. The baseline model remains as a fallback. Local checkpoint metadata
is validated before loading.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForAudioClassification,
    AutoProcessor,
)

BASE_MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"
WAVLM_CPU_MODEL_DIR = Path("checkpoints/voice_emotion_wavlm_cpu")
FINETUNED_MODEL_DIR = Path("checkpoints/voice_emotion_wavlm")
SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_STEP_SECONDS = 6
EXPECTED_LABELS = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}


def pool_hidden_states(hidden_states):
    values = np.asarray(hidden_states, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Expected non-empty [frames, hidden_dim] WavLM states.")
    return np.concatenate(
        [values.mean(axis=0), values.std(axis=0)]
    ).astype(np.float32, copy=False)


class VoiceEmotionRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_metadata = None
        self.embedding_model = None
        self.feature_extractor = None
        self.svm_pipeline = None
        self.model = None
        self.processor = None

        cpu_metadata_path = WAVLM_CPU_MODEL_DIR / "speechmind_metadata.json"
        cpu_head_path = WAVLM_CPU_MODEL_DIR / "voice_emotion_head.joblib"
        if cpu_metadata_path.is_file() and cpu_head_path.is_file():
            try:
                metadata = json.loads(cpu_metadata_path.read_text(encoding="utf-8"))
                is_valid = (
                    metadata.get("model_type") == "wavlm_frozen_embedding_svc"
                    and metadata.get("backbone") == "microsoft/wavlm-base-plus"
                    and metadata.get("sample_rate") == SAMPLE_RATE
                    and set(metadata.get("labels", [])) == EXPECTED_LABELS
                    and metadata.get("pooling") == "mean_std"
                    and metadata.get("feature_dimension") == 1536
                    and metadata.get("selected_kernel") in {"linear", "rbf"}
                )
                if is_valid:
                    import joblib
                    self.model_metadata = metadata
                    self.model_source = str(WAVLM_CPU_MODEL_DIR)
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                        metadata["backbone"]
                    )
                    self.embedding_model = AutoModel.from_pretrained(
                        metadata["backbone"]
                    ).to(self.device)
                    self.embedding_model.eval()
                    self.svm_pipeline = joblib.load(cpu_head_path)
                    print(
                        "[VoiceEmotion] Loaded CPU-trained WavLM/SVM checkpoint "
                        f"on {self.device}."
                    )
                    return
            except (OSError, ValueError, TypeError, ImportError, RuntimeError) as error:
                print(f"[VoiceEmotion] CPU WavLM checkpoint could not load: {error}")

        metadata_path = FINETUNED_MODEL_DIR / "speechmind_metadata.json"
        config_path = FINETUNED_MODEL_DIR / "config.json"
        if metadata_path.is_file() and config_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("model_type") == "wavlm_full_finetune"
                    and set(metadata.get("labels", [])) == EXPECTED_LABELS
                    and metadata.get("backbone") == "microsoft/wavlm-base-plus"
                    and metadata.get("sample_rate") == SAMPLE_RATE
                ):
                    self.model_source = str(FINETUNED_MODEL_DIR)
                    self.model_metadata = metadata
                else:
                    self.model_source = BASE_MODEL_NAME
            except (OSError, ValueError, TypeError):
                self.model_source = BASE_MODEL_NAME
        else:
            self.model_source = BASE_MODEL_NAME

        print(f"[VoiceEmotion] Loading baseline {self.model_source} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_source)
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_source
        ).to(self.device)
        self.model.eval()
        print("[VoiceEmotion] Baseline loaded. Labels:", self.model.config.id2label)

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
    def _wavlm_probabilities(self, chunk):
        encoded = self.feature_extractor(
            chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        inputs = encoded["input_values"].to(self.device)
        hidden = self.embedding_model(input_values=inputs).last_hidden_state[0]
        embedding = pool_hidden_states(hidden.cpu().numpy()).reshape(1, -1)
        probabilities = self.svm_pipeline.predict_proba(embedding)[0]
        classes = self.svm_pipeline.named_steps["classifier"].classes_
        return {str(label): float(prob) for label, prob in zip(classes, probabilities)}

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
        weighted = {}
        total_weight = 0.0
        for chunk in chunks:
            if self.svm_pipeline is not None:
                distribution = self._wavlm_probabilities(chunk)
            else:
                inputs = self.processor(
                    chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                logits = self.model(**inputs).logits[0]
                probabilities = torch.softmax(logits, dim=-1).float().cpu().numpy()
                id2label = self.model.config.id2label
                distribution = {
                    str(id2label.get(index, id2label.get(str(index), f"label_{index}"))).lower(): float(value)
                    for index, value in enumerate(probabilities)
                }
            weight = float(len(chunk))
            for label, probability in distribution.items():
                weighted[label] = weighted.get(label, 0.0) + probability * weight
            total_weight += weight

        distribution = {
            label: value / max(total_weight, 1.0)
            for label, value in weighted.items()
        }
        emotion = max(distribution, key=distribution.get)
        return {
            "emotion": emotion,
            "confidence": round(float(distribution[emotion]), 3),
            "distribution": distribution,
            "arousal": self._estimate_arousal(distribution),
            "inference_device": str(self.device),
            "inference_windows": len(chunks),
            "model_source": self.model_source,
            "validation_macro_f1": (
                self.model_metadata.get("validation_macro_f1")
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
