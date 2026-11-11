"""
SpeechMind local ASR with automatic Hindi-English code-switch probing.

Whisper supplies a primary-language estimate. Hindi-dominant and uncertain
English utterances may be checked by the isolated Srota provider; Srota is
selected only when its output visibly contains both Devanagari and Latin text.
Pure-Hindi and clear-English routes retain their existing specialist models.
"""

import torch
import whisper
from transformers import AutoModel

from models.asr_quality import estimate_whisper_quality
from models.srota_provider import SrotaASRProvider, has_mixed_devanagari_latin

WHISPER_MODEL = "small"
INDIC_CONFORMER_MODEL = "ai4bharat/indic-conformer-600m-multilingual"
MIN_INDIC_ROUTE_SCORE = 0.65
MIN_WHISPER_LANGUAGE_LOCK_SCORE = 0.70
# Srota probing is limited to Hindi-dominant or uncertain English utterances.
# The detector score is a routing heuristic, not a calibrated probability.
MIN_SROTA_PROBE_ENGLISH_SCORE = 0.90

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
        print(f"[ASR] Loading Whisper-{WHISPER_MODEL} for language identification...")
        self.whisper = whisper.load_model(WHISPER_MODEL, device=str(self.device))
        self.indic_model = None
        self.indic_model_error = None
        self.srota_provider = None
        self.srota_provider_error = None
        print(
            f"[ASR] Whisper loaded on {self.device}; "
            "IndicConformer remains lazy-loaded."
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

    def _try_srota_code_switch(self, audio_numpy, language_score=None):
        """Use Srota only when its output visibly preserves Hindi-English mixing."""
        try:
            if self.srota_provider is None:
                self.srota_provider = SrotaASRProvider()
            result = self.srota_provider.transcribe_audio(
                audio_numpy, sample_rate=16000
            )
            transcript = str(result.get("transcript") or "").strip()
            if not has_mixed_devanagari_latin(transcript):
                return None
            return {
                "transcript": transcript,
                "language": "hi",
                "confidence": None,
                "quality_score": None,
                "quality_method": "unavailable_for_srota",
                "language_detection_score": (
                    round(float(language_score), 3)
                    if language_score is not None else None
                ),
                "language_source": "whisper_detection_srota_code_switch_probe",
                "asr_model": result.get("asr_model", "qwen3-asr-0.6b-hinglish"),
                "asr_fallback": False,
                "asr_fallback_reason": None,
                "asr_candidate_model": "qwen3-asr-0.6b-hinglish",
                "asr_candidate_inference_seconds": result.get("inference_seconds"),
                "code_switch_detected": True,
            }
        except Exception as error:
            self.srota_provider_error = f"{type(error).__name__}: {error}"
            print(
                "[ASR] Srota code-switch probe unavailable; retaining the "
                f"existing route: {self.srota_provider_error}"
            )
            return None

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

        # Local open-source inference path. Language is never selected in the UI.
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

        # Probe Hindi-dominant speech with the isolated Hinglish specialist.
        # Only accept it when the transcript contains both Devanagari and Latin
        # script; pure-Hindi output stays on the existing IndicConformer route.
        if indic_language == "hi" and language_score >= MIN_INDIC_ROUTE_SCORE:
            srota_result = self._try_srota_code_switch(
                audio_numpy, language_score=language_score
            )
            if srota_result is not None:
                return srota_result

        # Whisper can classify a code-switched utterance as English. Probe only
        # uncertain English detections to avoid loading/running Srota on clear
        # English-only speech.
        if (
            detected_language in ("en", "english")
            and language_score < MIN_SROTA_PROBE_ENGLISH_SCORE
        ):
            srota_result = self._try_srota_code_switch(
                audio_numpy, language_score=language_score
            )
            if srota_result is not None:
                return srota_result

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
