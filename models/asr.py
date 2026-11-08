"""
SpeechMind - Speech Recognition + Language Identification

Architecture:

    Audio
      |
      v
    Whisper-small
      |
      +----> Language Identification
      |
      +----> English transcript (if English)
      |
      v
    IndicConformer
      |
      +----> Indian-language transcript

Whisper is used for language identification.
IndicConformer is used for transcription of supported
Indian languages.

IndicConformer supports 22 official Indian languages.
"""
from utils.ffmpeg_setup import *

import os

FFMPEG_BIN = r"C:\Users\kvasi\AppData\Local\Microsoft\WinGet\Packages\BtbN.FFmpeg.GPL.Shared.7.1_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-n7.1.5-12-g1fdbca85aa-win64-gpl-shared-7.1\bin"

if os.path.isdir(FFMPEG_BIN):
    os.add_dll_directory(FFMPEG_BIN)

import torch
import torchaudio

import whisper

from transformers import AutoModel


# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

WHISPER_MODEL = "small"

INDIC_CONFORMER_MODEL = (
    "ai4bharat/indic-conformer-600m-multilingual"
)


# ==========================================================
# LANGUAGE INFORMATION
# ==========================================================

INDIC_LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "kok": "Konkani",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}


# Whisper occasionally uses language names/codes that need
# normalisation before being passed to IndicConformer.

WHISPER_TO_INDIC = {
    "assamese": "as",
    "as": "as",

    "bengali": "bn",
    "bn": "bn",

    "bodo": "brx",
    "brx": "brx",

    "dogri": "doi",
    "doi": "doi",

    "gujarati": "gu",
    "gu": "gu",

    "hindi": "hi",
    "hi": "hi",

    "kannada": "kn",
    "kn": "kn",

    "konkani": "kok",
    "kok": "kok",

    "kashmiri": "ks",
    "ks": "ks",

    "maithili": "mai",
    "mai": "mai",

    "malayalam": "ml",
    "ml": "ml",

    "manipuri": "mni",
    "mni": "mni",

    "marathi": "mr",
    "mr": "mr",

    "nepali": "ne",
    "ne": "ne",

    "odia": "or",
    "oriya": "or",
    "or": "or",

    "punjabi": "pa",
    "pa": "pa",

    "sanskrit": "sa",
    "sa": "sa",

    "santali": "sat",
    "sat": "sat",

    "sindhi": "sd",
    "sd": "sd",

    "tamil": "ta",
    "ta": "ta",

    "telugu": "te",
    "te": "te",

    "urdu": "ur",
    "ur": "ur",
}


# ==========================================================
# SPEECH RECOGNIZER
# ==========================================================


class SpeechRecognizer:

    def __init__(self):

        print(
            "[ASR] Loading Whisper-small "
            "for language identification..."
        )

        # --------------------------------------------------
        # Whisper
        # --------------------------------------------------

        self.whisper = whisper.load_model(
            WHISPER_MODEL
        )

        print(
            "[ASR] Whisper-small loaded."
        )

        # --------------------------------------------------
        # IndicConformer
        # --------------------------------------------------

        print(
            "[ASR] Loading IndicConformer-600M..."
        )

        self.indic_model = AutoModel.from_pretrained(
            INDIC_CONFORMER_MODEL,
            trust_remote_code=True,
        )

        self.indic_model.eval()

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.indic_model.to(
            self.device
        )

        print(
            "[ASR] IndicConformer loaded."
        )

        print(
            "[ASR] Device:",
            self.device
        )

    # ======================================================
    # MAIN TRANSCRIPTION FUNCTION
    # ======================================================

    def transcribe(
        self,
        audio
    ) -> dict:

        """
        Parameters
        ----------
        audio:
            Preprocessed audio waveform.

        Returns
        -------
        dict

        {
            "transcript": str,
            "language": str,
            "confidence": float,
            "asr_model": str
        }
        """

        # --------------------------------------------------
        # Convert audio to numpy for Whisper
        # --------------------------------------------------

        if isinstance(
            audio,
            torch.Tensor
        ):

            audio_numpy = (
                audio
                .detach()
                .cpu()
                .numpy()
            )

        else:

            audio_numpy = audio

        # Remove unnecessary dimensions.

        audio_numpy = audio_numpy.squeeze()

        # --------------------------------------------------
        # Stage 1:
        # Whisper language identification
        # --------------------------------------------------

        print(
            "[ASR] Detecting language with Whisper..."
        )

        whisper_result = self.whisper.transcribe(
            audio_numpy,
            task="transcribe",
            fp16=False,
            verbose=False,
        )

        detected_language = (
            whisper_result
            .get("language", "unknown")
            .lower()
            .strip()
        )

        print(
            "[ASR] Whisper detected:",
            detected_language
        )

        # --------------------------------------------------
        # Normalize language code
        # --------------------------------------------------

        indic_language = WHISPER_TO_INDIC.get(
            detected_language
        )

        # --------------------------------------------------
        # Stage 2A:
        # English
        #
        # IndicConformer is intended for the 22 Indian
        # languages. For English, keep Whisper.
        # --------------------------------------------------

        if detected_language in (
            "en",
            "english",
        ):

            transcript = whisper_result.get(
                "text",
                ""
            ).strip()

            print(
                "[ASR] English detected."
            )

            print(
                "[ASR] Using Whisper transcript."
            )

            return {
                "transcript": transcript,
                "language": "en",
                "confidence": self._estimate_whisper_confidence(
                    whisper_result
                ),
                "asr_model": "whisper-small",
            }

        # --------------------------------------------------
        # Stage 2B:
        # Indian language
        # --------------------------------------------------

        if indic_language in INDIC_LANGUAGES:

            print(
                "[ASR] Indian language detected:",
                INDIC_LANGUAGES[indic_language],
            )

            print(
                "[ASR] Running IndicConformer..."
            )

            transcript = self._indic_transcribe(
                audio,
                indic_language,
            )

            return {
                "transcript": transcript,
                "language": indic_language,
                "confidence": self._estimate_indic_confidence(
                    transcript
                ),
                "asr_model": "indic-conformer-600m",
            }

        # --------------------------------------------------
        # Fallback
        #
        # If Whisper detects a language that isn't in our
        # supported Indic list, retain Whisper's transcript.
        # --------------------------------------------------

        print(
            "[ASR] Language is outside the "
            "IndicConformer language set."
        )

        print(
            "[ASR] Falling back to Whisper."
        )

        transcript = whisper_result.get(
            "text",
            ""
        ).strip()

        return {
            "transcript": transcript,
            "language": detected_language,
            "confidence": self._estimate_whisper_confidence(
                whisper_result
            ),
            "asr_model": "whisper-small",
        }

    # ======================================================
    # INDICCONFORMER TRANSCRIPTION
    # ======================================================

    def _indic_transcribe(
        self,
        audio,
        language_code,
    ):

        # --------------------------------------------------
        # Convert to Torch tensor
        # --------------------------------------------------

        if isinstance(
            audio,
            torch.Tensor
        ):

            wav = audio.detach().clone()

        else:

            wav = torch.tensor(
                audio,
                dtype=torch.float32,
            )

        # --------------------------------------------------
        # Ensure mono
        # --------------------------------------------------

        wav = wav.squeeze()

        # --------------------------------------------------
        # Add batch dimension
        #
        # IndicConformer expects:
        # [batch, time]
        # --------------------------------------------------

        wav = wav.unsqueeze(0)

        wav = wav.to(
            self.device
        )

        # --------------------------------------------------
        # CTC decoding
        #
        # AI4Bharat documents:
        #
        # model(wav, "hi", "ctc")
        #
        # where the language code is supplied explicitly.
        # --------------------------------------------------

        with torch.no_grad():

            transcription = self.indic_model(
                wav,
                language_code,
                "ctc",
            )

        # --------------------------------------------------
        # Normalize model output
        # --------------------------------------------------

        if isinstance(
            transcription,
            (list, tuple)
        ):

            transcription = transcription[0]

        transcript = str(
            transcription
        ).strip()

        return transcript

    # ======================================================
    # CONFIDENCE
    # ======================================================

    @staticmethod
    def _estimate_whisper_confidence(
        whisper_result
    ):

        """
        Whisper does not provide a single standardized
        probability for the complete transcription.

        We therefore derive a conservative proxy from
        segment-level average log probabilities.

        This is a heuristic, not a calibrated probability.
        """

        segments = whisper_result.get(
            "segments",
            []
        )

        if not segments:

            return 0.5

        values = []

        for segment in segments:

            avg_logprob = segment.get(
                "avg_logprob"
            )

            if avg_logprob is not None:

                values.append(
                    avg_logprob
                )

        if not values:

            return 0.5

        average = sum(values) / len(
            values
        )

        # Map log probability approximately to
        # a 0-1 range for downstream weighting.

        confidence = torch.sigmoid(
            torch.tensor(
                average
            )
        ).item()

        return round(
            float(confidence),
            3
        )

    @staticmethod
    def _estimate_indic_confidence(
        transcript
    ):

        """
        IndicConformer's simple inference call does not
        expose a calibrated confidence value through the
        interface we are using.

        Until we implement token-level CTC probabilities,
        return a conservative heuristic.

        This should NOT be described as a calibrated ASR
        probability in the final research evaluation.
        """

        if not transcript:

            return 0.0

        # Non-empty transcription gets a neutral baseline.
        return 0.7