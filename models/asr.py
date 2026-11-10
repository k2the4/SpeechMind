"""
SpeechMind ASR and language identification.

The default local provider retains Whisper/IndicConformer compatibility.
An optional Bodhan API provider is available for code-mixed Indian speech.
The hosted provider is opt-in because it sends audio to a third party and is
metered. Whisper's language detector supplies only a primary-language hint;
it does not describe all language switches in the utterance.
"""
import os

import torch
import whisper
from transformers import AutoModel

from models.asr_quality import estimate_whisper_quality
from models.bodhan_asr import BodhanASR

WHISPER_MODEL = "small"
INDIC_CONFORMER_MODEL = "ai4bharat/indic-conformer-600m-multilingual"
MIN_INDIC_ROUTE_SCORE = 0.65
MIN_WHISPER_LANGUAGE_LOCK_SCORE = 0.70

INDIC_LANGUAGES = {
    "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri",
    "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "kok": "Konkani",
    "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam", "mni": "Manipuri",
    "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
    "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu",
}
WHISPER_TO_INDIC = {
    "assamese": "as", "as": "as", "bengali": "bn", "bn": "bn",
    "bodo": "brx", "brx": "brx", "dogri": "doi", "doi": "doi",
    "gujarati": "gu", "gu": "gu", "hindi": "hi", "hi": "hi",
    "kannada": "kn", "kn": "kn", "konkani": "kok", "kok": "kok",
    "kashmiri": "ks", "ks": "ks", "maithili": "mai", "mai": "mai",
    "malayalam": "ml", "ml": "ml", "manipuri": "mni", "mni": "mni",
    "marathi": "mr", "mr": "mr", "nepali": "ne", "ne": "ne",
    "odia": "or", "oriya": "or", "or": "or", "punjabi": "pa", "pa": "pa",
    "sanskrit": "sa", "sa": "sa", "santali": "sat", "sat": "sat",
    "sindhi": "sd", "sd": "sd", "tamil": "ta", "ta": "ta",
    "telugu": "te", "te": "te", "urdu": "ur", "ur": "ur",
}
LANGUAGE_ALIASES = {
    "english": "en", "assamese": "as", "bengali": "bn", "gujarati": "gu",
    "hindi": "hi", "kannada": "kn", "malayalam": "ml", "marathi": "mr",
    "odia": "or", "punjabi": "pa", "tamil": "ta", "telugu": "te",
    "urdu": "ur",
}


class SpeechRecognizer:
    def __init__(self):
        self.model_name = f"openai/whisper-{WHISPER_MODEL}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        provider = os.getenv("SPEECHMIND_ASR_PROVIDER", "local").strip().lower()
        if provider not in {"local", "bodhan"}:
            raise ValueError(
                "SPEECHMIND_ASR_PROVIDER must be either 'local' or 'bodhan'."
            )
        self.bodhan_asr = BodhanASR() if provider == "bodhan" else None
        print(f"[ASR] Loading Whisper-{WHISPER_MODEL} for language identification...")
        self.whisper = whisper.load_model(WHISPER_MODEL, device=str(self.device))
        self.indic_model = None
        self.indic_model_error = None
        print(
            f"[ASR] Whisper loaded on {self.device}; "
            f"ASR provider={provider}; IndicConformer remains lazy-loaded."
        )

    def _detect_language(self, audio_numpy):
        """Return Whisper's primary-language guess and uncalibrated score."""
        padded = whisper.pad_or_trim(audio_numpy)
        mel = whisper.log_mel_spectrogram(
            padded, n_mels=self.whisper.dims.n_mels
        ).to(self.device)
        _, probabilities = self.whisper.detect_language(mel)
        language = max(probabilities, key=probabilities.get)
        return str(language).lower().strip(), float(probabilities[language])

    def _load_indic_model(self):
        if self.indic_model_error is not None:
            raise RuntimeError(
                "IndicConformer initialization previously failed; using Whisper fallback."
            ) from self.indic_model_error
        if self.indic_model is None:
            print("[ASR] Loading IndicConformer-600M on demand...")
            try:
                self.indic_model = AutoModel.from_pretrained(
                    INDIC_CONFORMER_MODEL, trust_remote_code=True
                )
                self.indic_model.eval()
                self.indic_model.to(self.device)
            except Exception as error:
                self.indic_model_error = error
                raise
        return self.indic_model

    def _indic_result_with_whisper_fallback(
        self, audio_numpy, language_code, language_detection_score=None
    ):
        """Prefer IndicConformer; fall back to Whisper instead of aborting ASR."""
        try:
            transcript = self._indic_transcribe(audio_numpy, language_code)
            return {
                "transcript": transcript,
                "language": language_code,
                "confidence": None,
                "quality_score": None,
                "quality_method": "unavailable_for_indicconformer",
                "language_detection_score": (
                    round(language_detection_score, 3)
                    if language_detection_score is not None else None
                ),
                "language_source": "whisper_detection",
                "asr_model": "indic-conformer-600m",
                "asr_fallback": False,
                "asr_fallback_reason": None,
            }
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            print(
                "[ASR] IndicConformer failed; falling back to Whisper "
                f"for {language_code}: {reason}"
            )
            whisper_result = self._decode_with_whisper(
                audio_numpy, language=language_code
            )
            result = self._whisper_result(
                whisper_result,
                language_fallback=language_code,
                language_detection_score=language_detection_score,
                language_source="indicconformer_failed_whisper_fallback",
            )
            result["asr_fallback"] = True
            result["asr_fallback_reason"] = reason
            result["asr_requested_model"] = "indic-conformer-600m"
            return result

    def _decode_with_whisper(self, audio_numpy, language=None):
        return self.whisper.transcribe(
            audio_numpy,
            task="transcribe",
            language=language,
            fp16=False,
            verbose=False,
            beam_size=5,
            condition_on_previous_text=False,
        )

    def transcribe(self, audio, language_hint=None) -> dict:
        if isinstance(audio, torch.Tensor):
            audio_numpy = audio.detach().cpu().numpy()
        else:
            audio_numpy = audio
        audio_numpy = audio_numpy.squeeze()
        if audio_numpy.ndim != 1 or audio_numpy.size == 0:
            raise ValueError("ASR expects non-empty mono audio.")

        hint = None if language_hint is None else str(language_hint).lower().strip()
        if hint in ("", "auto", "auto-detect", "none"):
            hint = None
        hint = LANGUAGE_ALIASES.get(hint, hint)

        # The code-mixed provider must receive the complete utterance without a
        # user-supplied language lock. Whisper is used only to report a primary
        # language; Bodhan performs the actual transcription automatically.
        if self.bodhan_asr is not None:
            try:
                detected_language, language_score = self._detect_language(audio_numpy)
            except Exception as error:
                print(f"[ASR] Primary-language detection unavailable: {error}")
                detected_language, language_score = "unknown", None
            result = self.bodhan_asr.transcribe(audio_numpy)
            result["language"] = detected_language
            result["language_detection_score"] = (
                round(language_score, 3) if language_score is not None else None
            )
            result["language_source"] = (
                "whisper_primary_language_only;bodhan_code_mixed_asr"
            )
            result["language_hint_ignored"] = bool(hint)
            return result

        # Legacy local path remains available as the default and fallback.
        if hint:
            if hint in INDIC_LANGUAGES:
                result = self._indic_result_with_whisper_fallback(audio_numpy, hint)
                result["language_source"] = (
                    "user_selected_legacy_with_indicconformer_fallback"
                    if result.get("asr_fallback") else "user_selected_legacy"
                )
                return result
            whisper_result = self._decode_with_whisper(audio_numpy, language=hint)
            return self._whisper_result(
                whisper_result, language_fallback=hint, language_source="user_selected_legacy"
            )

        try:
            detected_language, language_score = self._detect_language(audio_numpy)
        except Exception as error:
            print(f"[ASR] Language detection failed; Whisper will detect during transcription: {error}")
            whisper_result = self._decode_with_whisper(audio_numpy)
            return self._whisper_result(
                whisper_result, language_source="whisper_transcription_auto"
            )

        indic_language = WHISPER_TO_INDIC.get(detected_language)
        if indic_language in INDIC_LANGUAGES and language_score >= MIN_INDIC_ROUTE_SCORE:
            print(f"[ASR] Detected {INDIC_LANGUAGES[indic_language]}; using IndicConformer.")
            return self._indic_result_with_whisper_fallback(
                audio_numpy, indic_language, language_detection_score=language_score
            )

        decode_language = (
            detected_language
            if language_score >= MIN_WHISPER_LANGUAGE_LOCK_SCORE
            else None
        )
        whisper_result = self._decode_with_whisper(
            audio_numpy, language=decode_language
        )
        return self._whisper_result(
            whisper_result,
            language_fallback=detected_language,
            language_detection_score=language_score,
            language_source="whisper_detection_and_decode",
        )

    @staticmethod
    def _whisper_result(
        whisper_result,
        language_fallback=None,
        language_detection_score=None,
        language_source="whisper_transcription_auto",
    ):
        transcript = whisper_result.get("text", "").strip()
        detected = str(
            whisper_result.get("language") or language_fallback or "unknown"
        ).lower().strip()
        language = "en" if detected in ("en", "english") else detected
        quality = estimate_whisper_quality(
            whisper_result.get("segments", []), transcript
        )
        result = {
            "transcript": transcript,
            "language": language,
            "confidence": quality["quality_score"],
            "quality_score": quality["quality_score"],
            "quality_method": "whisper_token_quality_heuristic_uncalibrated",
            "avg_logprob": quality["avg_logprob"],
            "avg_no_speech_prob": quality["avg_no_speech_prob"],
            "avg_compression_ratio": quality["avg_compression_ratio"],
            "asr_model": "whisper-small",
            "language_source": language_source,
        }
        if language_detection_score is not None:
            result["language_detection_score"] = round(
                float(language_detection_score), 3
            )
        return result

    def _indic_transcribe(self, audio, language_code):
        model = self._load_indic_model()
        if isinstance(audio, torch.Tensor):
            wav = audio.detach().clone().float()
        else:
            wav = torch.tensor(audio, dtype=torch.float32)
        wav = wav.squeeze().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            transcription = model(wav, language_code, "ctc")
        if isinstance(transcription, (list, tuple)):
            transcription = transcription[0]
        return str(transcription).strip()
