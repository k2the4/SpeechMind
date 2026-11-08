"""
SpeechMind - Voice Emotion Recognition

Direct Hugging Face wav2vec2 implementation.

Model:
    Dpngtm/wav2vec2-emotion-recognition

The model supports:
    angry
    disgust
    fear
    happy
    neutral
    sad
    surprise

This implementation intentionally does NOT use SpeechBrain,
k2, or Flair.
"""

import numpy as np
import torch

from transformers import (
    AutoProcessor,
    AutoModelForAudioClassification,
)


MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"
# Model card lists English as its training/inference language. Indian-language
# predictions are experimental until a Hindi/Indic evaluation set is used.
MODEL_LANGUAGE_SCOPE = "en"


class VoiceEmotionRecognizer:

    def __init__(self):

        print(
            "[VoiceEmotion] Loading wav2vec2 emotion model..."
        )

        self.device = torch.device("cpu")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForAudioClassification.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

        print(
            "[VoiceEmotion] Model loaded successfully."
        )

        print(
            "[VoiceEmotion] Labels:",
            self.model.config.id2label
        )

    # ======================================================
    # PREDICTION
    # ======================================================

    def predict(self, audio):

        """
        Parameters
        ----------
        audio:
            Preprocessed audio waveform.

        Expected:
            mono audio sampled at 16 kHz.

        Returns
        -------
        dict
        """

        # --------------------------------------------------
        # Convert audio to numpy
        # --------------------------------------------------

        if isinstance(audio, torch.Tensor):

            audio = audio.detach().cpu().numpy()

        audio = np.asarray(
            audio,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Remove unnecessary dimensions
        # --------------------------------------------------

        audio = np.squeeze(audio)

        # --------------------------------------------------
        # Prevent empty audio
        # --------------------------------------------------

        if audio.size == 0:

            raise ValueError(
                "Voice emotion model received empty audio."
            )

        # --------------------------------------------------
        # Normalize if necessary
        # --------------------------------------------------

        max_value = np.max(
            np.abs(audio)
        )

        if max_value > 1.0:

            audio = audio / max_value

        # --------------------------------------------------
        # Prepare input
        #
        # The model expects 16 kHz audio.
        # audio_utils.py should already perform resampling.
        # --------------------------------------------------

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # --------------------------------------------------
        # Inference
        # --------------------------------------------------

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        # --------------------------------------------------
        # Convert logits → probabilities
        # --------------------------------------------------

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        # --------------------------------------------------
        # Build full probability distribution
        # --------------------------------------------------

        distribution = {}

        for index, probability in enumerate(
            probabilities
        ):

            label = self.model.config.id2label[
                index
            ]

            label = label.lower()

            distribution[label] = float(
                probability.item()
            )

        # --------------------------------------------------
        # Top emotion
        # --------------------------------------------------

        top_index = int(
            torch.argmax(probabilities).item()
        )

        emotion = self.model.config.id2label[
            top_index
        ].lower()

        confidence = float(
            probabilities[top_index].item()
        )

        # --------------------------------------------------
        # Estimate qualitative arousal
        #
        # This is NOT a medical measurement.
        # It is simply a prototype interpretation of
        # acoustic-emotion categories.
        # --------------------------------------------------

        arousal = self._estimate_arousal(
            distribution
        )

        return {

            "emotion": emotion,

            "confidence": round(
                confidence,
                3
            ),

            "distribution": distribution,

            "arousal": arousal,
        }

    # ======================================================
    # AROUSAL ESTIMATION
    # ======================================================

    @staticmethod
    def _estimate_arousal(
        distribution
    ):

        """
        Produces a qualitative arousal estimate.

        High-arousal emotions:
            angry, fear, happy, surprise

        Lower-arousal emotions:
            sad, neutral

        Disgust:
            moderate

        This is a prototype feature and should not be
        described as physiological arousal measurement.
        """

        high_arousal = (
            distribution.get("angry", 0.0)
            + distribution.get("fear", 0.0)
            + distribution.get("happy", 0.0)
            + distribution.get("surprise", 0.0)
        )

        low_arousal = (
            distribution.get("sad", 0.0)
            + distribution.get("neutral", 0.0)
        )

        if high_arousal >= 0.60:

            return "High"

        if low_arousal >= 0.60:

            return "Low"

        return "Moderate"