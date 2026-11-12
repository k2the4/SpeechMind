"""
SpeechMind end-to-end local inference pipeline.

The original transcript, English meaning, text analysis, voice analysis, and
combined interpretation remain distinct outputs. Optional analysis branches
fail independently so one model failure does not discard other valid results.
"""
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from utils.audio_utils import preprocess
from models.asr import SpeechRecognizer
from models.text_emotion import TextEmotionClassifier
from models.voice_emotion import VoiceEmotionRecognizer
from models.translation_routing import resolve_translation_route
from models.text_emotion_routing import resolve_text_emotion_language
from fusion_layer import fuse

TRANSLATABLE_LANGUAGES = {
    "hi", "bn", "ta", "te", "mr", "gu", "pa", "ml", "kn", "ur"
}


def _unavailable_branch(reason: str) -> dict:
    return {
        "emotion": "unavailable",
        "confidence": 0.0,
        "distribution": {},
        "secondary_emotions": [],
        "available": False,
        "reason": reason,
        "model_source": None,
    }


class SpeechMindPipeline:
    def __init__(self, load_translator: bool = True):
        print("[Pipeline] Initializing SpeechMind models...")
        self.component_errors = {}
        self.asr = self._initialize_component("asr", SpeechRecognizer)
        self.text_emotion = self._initialize_component(
            "text_emotion", TextEmotionClassifier
        )
        self.voice_emotion = self._initialize_component(
            "voice_emotion", VoiceEmotionRecognizer
        )
        self.translator = None
        self.translator_enabled = load_translator
        self._translator_lock = threading.Lock()
        print(
            "[Pipeline] Model initialization complete. "
            "Unavailable components will be reported independently."
        )

    def _initialize_component(self, name, factory):
        try:
            return factory()
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            self.component_errors[name] = reason
            print(f"[Pipeline] {name} initialization unavailable: {reason}")
            return None

    def _get_translator(self):
        # Streamlit can serve repeated requests through one cached pipeline.
        # Initialize the isolated worker once even if two requests arrive together.
        with self._translator_lock:
            if self.translator is None:
                print("[Pipeline] Initializing isolated IndicTrans2 translator...")
                from models.translation import Translator
                self.translator = Translator()
            return self.translator

    def _translate(self, transcript: str, asr_result: dict) -> dict:
        started = time.perf_counter()
        route = resolve_translation_route(
            detected_language=asr_result.get("language"),
            language_spans=asr_result.get("language_spans"),
            romanized_hindi_detected=asr_result.get(
                "romanized_hindi_detected", False
            ),
        )
        result = {
            "text": transcript,
            "available": False,
            "source": None,
            "source_language": route["source_language"],
            "method": route["method"],
            "note": route["note"],
            "error": route["error"],
            "duration_sec": 0.0,
        }
        try:
            if not transcript:
                result["error"] = "No transcript was produced."
            elif not route["available"]:
                # Do not call a Hindi-only MT model with Romanized Hindi and
                # display its output as if meaning preservation were supported.
                pass
            elif route["source_language"] == "en":
                result.update(
                    text=transcript,
                    available=True,
                    source="original_english_transcript",
                )
            elif not self.translator_enabled:
                result["error"] = "Translation is disabled."
            else:
                translator = self._get_translator()
                translated = translator.to_english(
                    transcript, route["source_language"]
                )
                result.update(
                    text=translated,
                    available=True,
                    source=translator.model_name,
                )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            print(f"[Pipeline] Translation unavailable: {result['error']}")
        result["duration_sec"] = round(time.perf_counter() - started, 3)
        return result

    def _predict_text_emotion(
        self, transcript: str, language: str, asr_result: dict = None
    ) -> dict:
        if not transcript.strip():
            return _unavailable_branch(
                "Text analysis skipped because no original transcript is available."
            )
        if self.text_emotion is None:
            return _unavailable_branch(
                self.component_errors.get(
                    "text_emotion", "Text-emotion model is unavailable."
                )
            )

        asr_result = asr_result if isinstance(asr_result, dict) else {}
        route = resolve_text_emotion_language(
            detected_language=language,
            languages_present=asr_result.get("languages_present"),
            code_switch_detected=bool(asr_result.get("code_switch_detected", False)),
            mixed_script_detected=bool(asr_result.get("mixed_script_detected", False)),
            romanized_hindi_detected=bool(
                asr_result.get("romanized_hindi_detected", False)
            ),
            supported_languages=getattr(
                self.text_emotion, "supported_languages", set()
            ),
        )
        if not route["available"]:
            result = _unavailable_branch(route["reason"])
            result["input_language"] = route["language"]
            result["routing_note"] = route["reason"]
            return result

        try:
            # Deliberately use the original transcript, never the English meaning.
            result = self.text_emotion.predict(transcript, route["language"])
            if not isinstance(result, dict):
                result = _unavailable_branch(
                    "Text-emotion model returned an invalid result."
                )
            result["input_language"] = route["language"]
            result["routing_note"] = route["reason"]
            return result
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            print(f"[Pipeline] Text-emotion analysis unavailable: {reason}")
            result = _unavailable_branch(reason)
            result["input_language"] = route["language"]
            return result

    def _predict_voice_emotion(self, audio) -> dict:
        if self.voice_emotion is None:
            return _unavailable_branch(
                self.component_errors.get(
                    "voice_emotion", "Voice-emotion model is unavailable."
                )
            )
        try:
            # This branch receives the prepared waveform directly, independent of ASR.
            result = self.voice_emotion.predict(audio)
            return result if isinstance(result, dict) else _unavailable_branch(
                "Voice-emotion model returned an invalid result."
            )
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            print(f"[Pipeline] Voice-emotion analysis unavailable: {reason}")
            return _unavailable_branch(reason)

    def run(
        self, audio_path: str, language_hint: str = None, asr_context: str = None
    ) -> dict:
        total_start = time.perf_counter()
        timings = {}

        stage_start = time.perf_counter()
        audio = preprocess(audio_path)
        timings["preprocessing"] = round(time.perf_counter() - stage_start, 3)

        stage_start = time.perf_counter()
        asr_error = None
        if self.asr is None:
            asr_error = self.component_errors.get(
                "asr", "Speech-recognition model is unavailable."
            )
            asr_result = {}
        else:
            try:
                if asr_context and asr_context.strip():
                    asr_result = self.asr.transcribe(
                        audio, language_hint=language_hint, context=asr_context
                    )
                else:
                    asr_result = self.asr.transcribe(
                        audio, language_hint=language_hint
                    )
                if not isinstance(asr_result, dict):
                    raise TypeError("ASR returned an invalid result.")
            except Exception as error:
                asr_error = f"{type(error).__name__}: {error}"
                print(f"[Pipeline] ASR unavailable for this clip: {asr_error}")
                asr_result = {}
        timings["asr"] = round(time.perf_counter() - stage_start, 3)

        transcript = str(asr_result.get("transcript") or "").strip()
        language = str(asr_result.get("language") or "unknown").lower().strip()
        quality_value = asr_result.get("quality_score")
        if quality_value is None:
            quality_value = asr_result.get("confidence")
        try:
            asr_confidence = (
                max(0.0, min(1.0, float(quality_value)))
                if quality_value is not None else None
            )
        except (TypeError, ValueError):
            asr_confidence = None

        asr_quality_status = (
            "unavailable" if asr_confidence is None
            else "uncalibrated_estimate"
        )
        asr_quality_note = (
            "The selected recognizer does not expose a usable transcript-quality "
            "score. Check the original transcript before relying on its English "
            "meaning or text-emotion result."
            if asr_confidence is None else
            "The transcript-quality value is an uncalibrated model heuristic, "
            "not a probability that every recognised word is correct."
        )

        stage_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            text_future = executor.submit(
                self._predict_text_emotion, transcript, language, asr_result
            )
            voice_future = executor.submit(self._predict_voice_emotion, audio)
            translation_future = executor.submit(
                self._translate, transcript, asr_result
            )
            text_result = text_future.result()
            voice_result = voice_future.result()
            translation_result = translation_future.result()
        timings["parallel_analysis_and_translation"] = round(
            time.perf_counter() - stage_start, 3
        )
        timings["translation"] = translation_result["duration_sec"]

        stage_start = time.perf_counter()
        fused = fuse(
            text_result=text_result,
            voice_result=voice_result,
            asr_confidence=asr_confidence,
        )
        timings["fusion"] = round(time.perf_counter() - stage_start, 3)

        english_meaning = translation_result["text"]
        if translation_result["available"]:
            existing_note = translation_result.get("note")
            translation_result["note"] = (
                f"{existing_note} {asr_quality_note}"
                if existing_note else asr_quality_note
            )
        timings["total"] = round(time.perf_counter() - total_start, 3)

        return {
            "language": language,
            "language_detection_score": asr_result.get("language_detection_score"),
            "language_source": asr_result.get("language_source"),
            "transcript": transcript,
            "asr_model": asr_result.get("asr_model", "unavailable"),
            "asr_provider": "local",
            "asr_error": asr_error,
            "asr_fallback": bool(asr_result.get("asr_fallback", False)),
            "asr_fallback_reason": asr_result.get("asr_fallback_reason"),
            "asr_requested_model": asr_result.get("asr_requested_model"),
            "code_switch_detected": bool(asr_result.get("code_switch_detected", False)),
            "language_spans": asr_result.get("language_spans", []),
            "language_evidence": asr_result.get("language_evidence", {}),
            "script_evidence": asr_result.get("script_evidence", {}),
            "languages_present": asr_result.get("languages_present", []),
            "scripts_present": asr_result.get("scripts_present", []),
            "mixed_script_detected": bool(asr_result.get("mixed_script_detected", False)),
            "romanized_hindi_detected": bool(asr_result.get("romanized_hindi_detected", False)),
            "language_span_unit": asr_result.get("span_unit"),
            "audio_timestamps_available": bool(asr_result.get("audio_timestamps_available", False)),
            "asr_candidate_model": asr_result.get("asr_candidate_model"),
            "asr_candidate_inference_seconds": asr_result.get(
                "asr_candidate_inference_seconds"
            ),
            "asr_context_terms": asr_result.get("asr_context_terms", ""),
            "asr_context_source": asr_result.get("asr_context_source", "none"),
            "asr_context_probe_seconds": asr_result.get(
                "asr_context_probe_seconds", 0.0
            ),
            "asr_context_probe_transcript": asr_result.get(
                "asr_context_probe_transcript", ""
            ),
            "asr_inference_device": asr_result.get("asr_inference_device"),
            "asr_confidence": round(asr_confidence, 3) if asr_confidence is not None else None,
            "asr_quality_score": (
                round(asr_confidence, 3) if asr_confidence is not None else None
            ),
            "asr_quality_status": asr_quality_status,
            "asr_quality_note": asr_quality_note,
            "asr_quality_method": asr_result.get("quality_method"),
            "asr_avg_logprob": asr_result.get("avg_logprob"),
            "asr_avg_no_speech_prob": asr_result.get("avg_no_speech_prob"),
            "asr_avg_compression_ratio": asr_result.get("avg_compression_ratio"),
            "meaning_en": english_meaning,
            "meaning_en_available": translation_result["available"],
            "meaning_en_source": translation_result["source"],
            "meaning_en_source_language": translation_result["source_language"],
            "meaning_en_method": translation_result["method"],
            "meaning_en_note": translation_result["note"],
            "text_emotion": fused["text_emotion"],
            "text_emotion_available": bool(text_result.get("available", False)),
            "text_emotion_reason": text_result.get("reason"),
            "text_emotion_model_source": text_result.get("model_source"),
            "text_emotion_input_language": text_result.get("input_language"),
            "text_emotion_routing_note": text_result.get("routing_note"),
            "text_emotion_secondary": text_result.get("secondary_emotions", []),
            "text_emotion_multi_label_scores": text_result.get("multi_label_scores"),
            "text_emotion_task_type": text_result.get("task_type"),
            "text_emotion_languages": sorted(
                getattr(self.text_emotion, "supported_languages", []) or []
            ),
            "voice_emotion": fused["voice_emotion"],
            "voice_emotion_available": bool(voice_result.get("available", True)),
            "voice_emotion_reason": voice_result.get("reason"),
            "voice_model_source": voice_result.get("model_source"),
            "voice_inference_device": voice_result.get("inference_device"),
            "voice_inference_windows": voice_result.get("inference_windows"),
            "voice_validation_macro_f1": voice_result.get("validation_macro_f1"),
            "valence": fused["valence"],
            "arousal": fused["arousal"],
            "arousal_available": fused["arousal_available"],
            "overall_interpretation": fused["overall_interpretation"],
            "confidence": fused["confidence"],
            "confidence_margin": fused["confidence_margin"],
            "confidence_is_calibrated": False,
            "low_confidence_warning": fused["low_confidence_warning"],
            "overall_emotion": fused["overall_emotion"],
            "fusion_weights": fused["fusion_weights"],
            "text_branch_available": fused["text_branch_available"],
            "voice_branch_available": fused["voice_branch_available"],
            "branch_agreement": fused["branch_agreement"],
            "branch_disagreement": fused["branch_disagreement"],
            "voice_label_ambiguous": fused.get("voice_label_ambiguous", False),
            "branch_comparison": fused["branch_comparison"],
            "interpretation_status": fused["interpretation_status"],
            "uncertainty_reasons": fused["uncertainty_reasons"],
            "text_emotion_distribution": fused["text_distribution"],
            "voice_emotion_distribution": fused["voice_distribution"],
            "fused_emotion_distribution": fused["fused_distribution"],
            "processing_time_sec": timings["total"],
            "timings_sec": timings,
            "translator_provider": (
                getattr(self.translator, "provider_name", "unknown")
                if self.translator is not None
                else ("disabled" if not self.translator_enabled else "not_initialized")
            ),
            "translator_loaded": bool(
                self.translator is not None and getattr(self.translator, "loaded", False)
            ),
            "translator_inference_seconds": (
                getattr(self.translator, "last_inference_seconds", None)
                if self.translator is not None else None
            ),
            "translator_error": translation_result["error"],
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python pipeline.py <path_to_audio_file>")
    pipeline = SpeechMindPipeline()
    for key, value in pipeline.run(sys.argv[1]).items():
        print(f"{key:28s}: {value}")
