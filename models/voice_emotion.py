"""SpeechMind voice-emotion inference with multi-corpus checkpoint preference."""
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

from models.voice_features import (
    POOLING_VERSION,
    EXPECTED_FEATURE_DIMENSION,
    extract_wavlm_embedding,
    rank_top_two_scores,
)

BASE_MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"
MULTICORPUS_MODEL_DIR = Path("checkpoints/voice_emotion_multicorpus_cpu")
WAVLM_CPU_MODEL_DIR = Path("checkpoints/voice_emotion_wavlm_cpu")
FINETUNED_MODEL_DIR = Path("checkpoints/voice_emotion_wavlm")
SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_STEP_SECONDS = 6
EXPECTED_LABELS = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}


def pool_hidden_states(hidden_states):
    values = np.asarray(hidden_states, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Expected non-empty [frames, hidden_size] states.")
    return np.concatenate(
        [
            values.mean(0),
            values.std(0),
            np.quantile(values, 0.1, axis=0),
            np.quantile(values, 0.9, axis=0),
        ]
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
        self.model_source = None

        options = [
            (
                MULTICORPUS_MODEL_DIR,
                "wavlm_multicorpus_feature_svc",
                POOLING_VERSION,
                EXPECTED_FEATURE_DIMENSION,
            ),
            (
                WAVLM_CPU_MODEL_DIR,
                "wavlm_frozen_embedding_svc",
                "mean_std",
                1536,
            ),
        ]
        for directory, model_type, pooling, dimension in options:
            meta_path = directory / "speechmind_metadata.json"
            head_path = directory / "voice_emotion_head.joblib"
            if not (meta_path.is_file() and head_path.is_file()):
                continue
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                valid = (
                    metadata.get("model_type") == model_type
                    and metadata.get("backbone") == "microsoft/wavlm-base-plus"
                    and metadata.get("sample_rate") == SAMPLE_RATE
                    and set(metadata.get("labels", [])) == EXPECTED_LABELS
                    and metadata.get("pooling") == pooling
                    and metadata.get("feature_dimension") == dimension
                    and metadata.get("selected_kernel") in {"linear", "rbf"}
                )
                if not valid:
                    print(f"[VoiceEmotion] Ignoring incompatible checkpoint: {directory}")
                    continue

                import joblib

                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    metadata["backbone"]
                )
                self.embedding_model = AutoModel.from_pretrained(
                    metadata["backbone"]
                ).to(self.device)
                self.embedding_model.eval()
                self.svm_pipeline = joblib.load(head_path)
                self.model_metadata = metadata
                self.model_source = str(directory)
                self.pooling_version = pooling
                print(
                    f"[VoiceEmotion] Loaded {model_type} from {directory} "
                    f"on {self.device}."
                )
                return
            except (
                OSError,
                ValueError,
                TypeError,
                ImportError,
                RuntimeError,
            ) as error:
                print(f"[VoiceEmotion] Could not load {directory}: {error}")

        meta = FINETUNED_MODEL_DIR / "speechmind_metadata.json"
        config = FINETUNED_MODEL_DIR / "config.json"
        if meta.is_file() and config.is_file():
            try:
                metadata = json.loads(meta.read_text(encoding="utf-8"))
                valid = (
                    metadata.get("model_type") == "wavlm_full_finetune"
                    and set(metadata.get("labels", [])) == EXPECTED_LABELS
                    and metadata.get("backbone") == "microsoft/wavlm-base-plus"
                    and metadata.get("sample_rate") == SAMPLE_RATE
                )
                if valid:
                    self.model_source = str(FINETUNED_MODEL_DIR)
                    self.model_metadata = metadata
                else:
                    self.model_source = BASE_MODEL_NAME
            except (OSError, ValueError, TypeError):
                self.model_source = BASE_MODEL_NAME
        else:
            self.model_source = BASE_MODEL_NAME

        print(
            f"[VoiceEmotion] Loading baseline {self.model_source} "
            f"on {self.device}..."
        )
        self.processor = AutoProcessor.from_pretrained(self.model_source)
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_source
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _to_chunks(audio):
        window = WINDOW_SECONDS * SAMPLE_RATE
        step = WINDOW_STEP_SECONDS * SAMPLE_RATE
        if len(audio) <= window:
            return [audio]
        chunks = [audio[start : start + window] for start in range(0, len(audio), step)]
        if len(chunks) > 1 and len(chunks[-1]) < SAMPLE_RATE:
            chunks.pop()
        return chunks or [audio]

    @torch.inference_mode()
    def _wavlm_probabilities(self, chunk):
        if self.pooling_version == "mean_std":
            encoded = self.feature_extractor(
                chunk,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            inputs = encoded["input_values"].to(self.device)
            hidden = self.embedding_model(
                input_values=inputs
            ).last_hidden_state[0]
            states = hidden.float().cpu().numpy()
            embedding = np.concatenate(
                [states.mean(axis=0), states.std(axis=0)]
            ).astype(np.float32, copy=False).reshape(1, -1)
        else:
            embedding = extract_wavlm_embedding(
                self.embedding_model,
                self.feature_extractor,
                chunk,
                self.device,
            ).reshape(1, -1)

        probabilities = self.svm_pipeline.predict_proba(embedding)[0]
        classes = self.svm_pipeline.named_steps["classifier"].classes_
        return {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities)
        }

    @torch.inference_mode()
    def predict(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        waveform = np.asarray(audio, dtype=np.float32).squeeze()
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Voice emotion model expects non-empty mono audio.")
        if not np.isfinite(waveform).all():
            raise ValueError("Voice emotion model received NaN or infinite audio.")

        peak = float(np.max(np.abs(waveform)))
        if peak > 1.0:
            waveform = waveform / peak

        chunks = self._to_chunks(waveform)
        weighted = {}
        total = 0.0
        for chunk in chunks:
            if self.svm_pipeline is not None:
                distribution = self._wavlm_probabilities(chunk)
            else:
                inputs = self.processor(
                    chunk,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                logits = self.model(**inputs).logits[0]
                probabilities = torch.softmax(logits, dim=-1).float().cpu().numpy()
                id_to_label = self.model.config.id2label
                distribution = {
                    str(
                        id_to_label.get(
                            index,
                            id_to_label.get(str(index), f"label_{index}"),
                        )
                    ).lower(): float(probability)
                    for index, probability in enumerate(probabilities)
                }

            weight = float(len(chunk))
            for label, probability in distribution.items():
                weighted[label] = weighted.get(label, 0.0) + probability * weight
            total += weight

        distribution = {
            label: value / max(total, 1.0)
            for label, value in weighted.items()
        }
        ranked = sorted(
            distribution.items(), key=lambda item: item[1], reverse=True
        )
        emotion = ranked[0][0]
        top_two, top_two_margin = rank_top_two_scores(distribution)
        return {
            "available": True,
            "emotion": emotion,
            "top_two": top_two,
            "top_two_margin": round(top_two_margin, 4),
            "confidence": round(float(distribution[emotion]), 3),
            "distribution": distribution,
            "arousal": "Not assessed",
            "arousal_available": False,
            "inference_device": str(self.device),
            "inference_windows": len(chunks),
            "model_source": self.model_source,
            "validation_macro_f1": (
                self.model_metadata.get("validation_macro_f1")
                if self.model_metadata
                else None
            ),
        }
