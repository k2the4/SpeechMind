"""
SpeechMind local ASR with automatic Hindi-English and Romanized-Hindi routing.

Whisper supplies a primary-language estimate and, for English detections, a
transcript-level Hinglish signal. Hindi-dominant and uncertain English
utterances may be checked by the isolated Srota provider; high-confidence
English detections are checked only when the Whisper transcript contains
Hindi/Hinglish evidence. Pure-Hindi and clear-English routes retain their
specialists.
"""

import torch
import whisper
from transformers import AutoModel

from models.asr_quality import estimate_whisper_quality
from models.hinglish_context import (
    build_hinglish_context,
    infer_hinglish_context_from_audio_tail,
)
from models.srota_provider import SrotaASRProvider
from models.language_spans import annotate_language_spans
from models.asr_routing import (
    audio_language_suggests_hindi,
    srota_candidate_valid_for_route,
    srota_supports_duration,
    whisper_transcript_suggests_hinglish,
)

WHISPER_MODEL = "small"
INDIC_CONFORMER_MODEL = "ai4bharat/indic-conformer-600m-multilingual"
MIN_INDIC_ROUTE_SCORE = 0.65
MIN_WHISPER_LANGUAGE_LOCK_SCORE = 0.70
# This threshold controls the early Srota probe for uncertain English.
# High-confidence English is probed only when its Whisper transcript suggests
# Hinglish. The detector score is not a calibrated probability.
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

def _attach_language_spans(result: dict) -> dict:
    """Attach conservative transcript offsets without inventing audio timing."""
    annotation = annotate_language_spans(
        result.get("transcript", ""), result.get("language")
    )
    result.update(annotation)
    result["code_switch_detected"] = bool(
        annotation["code_switch_detected"] or annotation["mixed_script_detected"]
    )
    return result


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
        language_probabilities = {
            str(code).lower().strip(): float(score)
            for code, score in probabilities.items()
        }
        return (
            str(language).lower().strip(),
            float(probabilities[language]),
            language_probabilities,
        )

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
            return _attach_language_spans({
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
            })
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

    def _infer_srota_context(self, audio_numpy):
        """Use an auto-detected Whisper tail pass to scout known English terms."""
        context_info = infer_hinglish_context_from_audio_tail(
            audio_numpy, self.whisper, sample_rate=16000, tail_seconds=5.0
        )
        if context_info["context"]:
            print(
                "[ASR] Automatic Srota vocabulary context: "
                + context_info["context"]
            )
        elif context_info["source"] == "whisper_tail_no_allowlisted_terms":
            print(
                "[ASR] Whisper tail scout found no allow-listed technical "
                "vocabulary; using native Srota decoding."
            )
        else:
            print(
                "[ASR] Automatic context unavailable: "
                + str(context_info["source"])
            )
        return (
            context_info["context"],
            context_info["source"],
            context_info["probe_seconds"],
            context_info["transcript"],
        )

    def _try_srota_code_switch(
        self, audio_numpy, language_score=None, primary_language=None,
        probe_source="whisper_detection",
        context=None,
        context_source=None,
    ):
        """Probe Srota for mixed-script Hinglish and fully Romanized Hindi."""
        if self.srota_provider_error is not None:
            return None
        if not srota_supports_duration(
            int(audio_numpy.size),
            sample_rate=16000,
            max_seconds=SrotaASRProvider.MAX_CLIP_SECONDS,
        ):
            print(
                "[ASR] Skipping Srota for this clip only: current worker limit "
                "is 30 seconds. Existing specialist routing will handle it."
            )
            return None
        try:
            if self.srota_provider is None:
                self.srota_provider = SrotaASRProvider()
            if not self.srota_provider.available:
                self.srota_provider_error = (
                    "Srota environment is not installed; run "
                    "SETUP_HINGLISH_ASR_ENV.bat once."
                )
                print(f"[ASR] {self.srota_provider_error}")
                return None
            context_probe_seconds = 0.0
            context_probe_transcript = ""
            if context is None:
                (
                    context,
                    context_source,
                    context_probe_seconds,
                    context_probe_transcript,
                ) = self._infer_srota_context(audio_numpy)
            else:
                context = str(context).strip()
                context_source = context_source or (
                    "explicit_context" if context else "explicit_empty_context"
                )

            result = self.srota_provider.transcribe_audio(
                audio_numpy,
                sample_rate=16000,
                context=context or None,
            )
            transcript = str(result.get("transcript") or "").strip()
            annotation = annotate_language_spans(transcript, primary_language)
            use_srota = srota_candidate_valid_for_route(
                transcript,
                primary_language or "unknown",
                probe_source=probe_source,
            )
            if not use_srota:
                return None
            source_prefix = {
                "whisper_transcript": "whisper_transcript_srota",
                "audio_language": "whisper_audio_language_srota",
            }.get(probe_source, "whisper_detection_srota")
            if annotation["mixed_script_detected"]:
                source_suffix = "code_switch_probe"
            elif annotation["romanized_hindi_detected"]:
                source_suffix = "romanized_hindi"
            elif "hi" in annotation["languages_present"]:
                source_suffix = "hindi_script"
            else:
                source_suffix = "code_switch_probe"
            source = f"{source_prefix}_{source_suffix}"
            return _attach_language_spans({
                "transcript": transcript,
                "language": "hi",
                "confidence": None,
                "quality_score": None,
                "quality_method": "unavailable_for_srota",
                "language_detection_score": (
                    round(float(language_score), 3)
                    if language_score is not None else None
                ),
                "language_source": source,
                "asr_model": result.get("asr_model", "qwen3-asr-0.6b-hinglish"),
                "asr_fallback": False,
                "asr_fallback_reason": None,
                "asr_candidate_model": result.get(
                    "asr_model", SrotaASRProvider.MODEL_NAME
                ),
                "asr_candidate_inference_seconds": result.get("inference_seconds"),
                "asr_context_terms": context or "",
                "asr_context_source": context_source,
                "asr_context_probe_seconds": context_probe_seconds,
                "asr_context_probe_transcript": context_probe_transcript,
                "asr_inference_device": (
                    result.get("worker_response", {}).get("device")
                    if isinstance(result.get("worker_response"), dict) else None
                ),
                "code_switch_detected": bool(
                    annotation["code_switch_detected"] or annotation["mixed_script_detected"]
                ),
            })
        except Exception as error:
            # A single request failure must not disable Srota for every later
            # clip in this long-lived app session. Only a missing installation
            # is cached above; transient/model errors are logged per request.
            print(
                "[ASR] Srota probe failed for this clip; retaining the "
                f"existing route: {type(error).__name__}: {error}"
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

    def transcribe(self, audio, language_hint=None, context=None) -> dict:
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
            detected_language, language_score, language_probabilities = self._detect_language(
                audio_numpy
            )
        except Exception as error:
            print(f"[ASR] Language detection failed; Whisper will detect during transcription: {error}")
            whisper_result = self._decode_with_whisper(audio_numpy)
            return self._whisper_result(
                whisper_result, language_source="whisper_transcription_auto"
            )

        indic_language = WHISPER_TO_INDIC.get(detected_language)

        # Probe Hindi-dominant speech with the isolated Hinglish specialist.
        # Accept mixed-script or sufficiently strong Romanized-Hindi evidence;
        # pure-Hindi output stays on the existing IndicConformer route.
        if indic_language == "hi" and language_score >= MIN_INDIC_ROUTE_SCORE:
            srota_result = self._try_srota_code_switch(
                audio_numpy, language_score=language_score,
                primary_language=indic_language,
                context=context,
            )
            if srota_result is not None:
                return srota_result

        # Low-confidence English detections get a direct specialist probe.
        # For high-confidence English, decode Whisper first (this decode is
        # needed anyway), then probe only when its transcript contains Hindi /
        # Hinglish evidence. This catches confidently mislabelled Romanized
        # Hindi without paying the Srota cost for clear English speech.
        if detected_language in ("en", "english"):
            srota_already_probed = False
            if language_score < MIN_SROTA_PROBE_ENGLISH_SCORE:
                srota_result = self._try_srota_code_switch(
                    audio_numpy, language_score=language_score,
                    primary_language=detected_language,
                    context=context,
                )
                srota_already_probed = True
                if srota_result is not None:
                    return srota_result
            elif audio_language_suggests_hindi(
                language_probabilities,
                detected_language,
                audio_numpy.size / 16000.0,
            ):
                print(
                    "[ASR] Short English-detected clip has a secondary Hindi "
                    "audio-language signal; probing Srota."
                )
                srota_result = self._try_srota_code_switch(
                    audio_numpy,
                    language_score=language_score,
                    primary_language=detected_language,
                    probe_source="audio_language",
                    context=context,
                )
                srota_already_probed = True
                if srota_result is not None:
                    return srota_result

            decode_language = (
                detected_language
                if language_score >= MIN_WHISPER_LANGUAGE_LOCK_SCORE
                else None
            )
            whisper_result = self._whisper_result(
                self._decode_with_whisper(audio_numpy, language=decode_language),
                language_fallback=detected_language,
                language_detection_score=language_score,
                language_source="whisper_detection_and_decode",
            )
            if (
                not srota_already_probed
                and whisper_transcript_suggests_hinglish(
                    whisper_result.get("transcript", ""),
                    whisper_result.get("language") or detected_language,
                )
            ):
                srota_result = self._try_srota_code_switch(
                    audio_numpy, language_score=language_score,
                    primary_language=detected_language,
                    probe_source="whisper_transcript",
                    context=(
                        context if context is not None
                        else build_hinglish_context(
                            whisper_result.get("transcript", "")
                        ) or None
                    ),
                    context_source="whisper_full_transcript",
                )
                if srota_result is not None:
                    return srota_result
            return whisper_result

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
        return _attach_language_spans(result)

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
