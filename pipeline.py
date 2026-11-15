"""
SpeechMind end-to-end local inference pipeline.

The original transcript, English meaning, text analysis, voice analysis, and
combined interpretation remain distinct outputs. Optional analysis branches
fail independently so one model failure does not discard other valid results.
"""
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from utils.runtime_config import configure_torch_runtime

from utils.audio_utils import preprocess_with_quality
from models.language_spans import annotate_language_spans
from models.asr import SpeechRecognizer
from models.text_emotion import TextEmotionClassifier
from models.voice_emotion import VoiceEmotionRecognizer
from models.translation_routing import resolve_translation_route
from models.text_emotion_routing import resolve_text_emotion_language
from fusion_layer import fuse

TRANSLATABLE_LANGUAGES = {
    "hi", "bn", "ta", "te", "mr", "gu", "pa", "ml", "kn", "ur"
}




def _run_timed(function, *args, **kwargs):
    """Run one inference branch and return its own elapsed time."""
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, round(time.perf_counter() - started, 3)


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
        # Configure PyTorch when the application pipeline is constructed,
        # not while this module is imported. This avoids loading native
        # extensions inside temporary sys.modules test fixtures.
        self.torch_runtime = configure_torch_runtime()
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
        translation_text = transcript
        transliteration = {"available": None, "text": None, "replacements": 0,
                           "model": None, "error": None}
        route = resolve_translation_route(
            detected_language=asr_result.get("language"),
            language_spans=asr_result.get("language_spans"),
            romanized_hindi_detected=asr_result.get("romanized_hindi_detected", False),
        )
        if route["method"] == "romanized_hindi_transliteration_required":
            from models.romanized_hindi import transliterate_romanized_hindi
            transliteration = transliterate_romanized_hindi(
                transcript, asr_result.get("language_spans", [])
            )
            if transliteration["available"]:
                translation_text = transliteration["text"]
                converted = annotate_language_spans(translation_text, "hi")
                route = resolve_translation_route(
                    "hi", converted["language_spans"], romanized_hindi_detected=False
                )
                route["method"] = "indicxlit_then_indictrans2"
                route["note"] = (
                    "Romanized-Hindi words were transliterated locally for translation; "
                    "the original transcript remains unchanged. Any mixed Hindi-English "
                    "IndicTrans2 translation remains best-effort and is not yet evaluated."
                )
            else:
                route["error"] = transliteration["error"]
        result = {
            "text": transcript, "available": False, "source": None,
            "source_language": route["source_language"], "method": route["method"],
            "note": route["note"], "error": route["error"], "duration_sec": 0.0,
            "cache_used": None, "translation_input_text": translation_text,
            "transliteration_available": transliteration["available"],
            "transliterated_text": transliteration["text"],
            "transliteration_replacements": transliteration["replacements"],
            "transliteration_model": transliteration["model"],
            "transliteration_error": transliteration["error"],
        }
        try:
            if not transcript:
                result["error"] = "No transcript was produced."
            elif route["available"] and route["source_language"] == "en":
                result.update(text=transcript, available=True,
                              source="original_english_transcript")
            elif route["available"] and not self.translator_enabled:
                result["error"] = "Translation is disabled."
            elif route["available"]:
                translator = self._get_translator()
                translated = translator.to_english(translation_text, route["source_language"])
                result.update(text=translated, available=True, source=translator.model_name,
                              cache_used=getattr(translator, "last_cache_used", None))
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

    def prepare(self, audio_path: str, language_hint: str = None,
                asr_context: str = None) -> dict:
        """Decode and transcribe once; keep the prepared waveform for later edits."""
        timings = {}
        started = time.perf_counter()
        audio, quality = preprocess_with_quality(audio_path)
        timings["preprocessing"] = round(time.perf_counter() - started, 3)
        started = time.perf_counter()
        error = None
        if self.asr is None:
            error = self.component_errors.get("asr", "Speech-recognition model is unavailable.")
            result = {}
        else:
            try:
                if asr_context and asr_context.strip():
                    result = self.asr.transcribe(audio, language_hint=language_hint,
                                                 context=asr_context)
                else:
                    result = self.asr.transcribe(audio, language_hint=language_hint)
                if not isinstance(result, dict):
                    raise TypeError("ASR returned an invalid result.")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"[Pipeline] ASR unavailable for this clip: {error}")
                result = {}
        timings["asr"] = round(time.perf_counter() - started, 3)
        return {"audio": audio, "audio_quality": quality, "asr_result": result,
                "asr_error": error, "timings_sec": timings}

    def analyze_prepared(
        self, prepared: dict, transcript_override=None,
        include_emotion: bool = True, include_translation: bool = True,
    ) -> dict:
        """Analyze prepared audio; corrected text never triggers another ASR call."""
        if not isinstance(prepared, dict) or "audio" not in prepared:
            raise ValueError("Prepared SpeechMind audio is missing or invalid.")
        audio = prepared["audio"]
        audio_quality = prepared.get("audio_quality") or {}
        asr_result = dict(prepared.get("asr_result") or {})
        asr_error = prepared.get("asr_error")
        timings = dict(prepared.get("timings_sec") or {})
        manual_transcript = transcript_override is not None
        transcript = (str(transcript_override or "").strip() if manual_transcript
                      else str(asr_result.get("transcript") or "").strip())
        if manual_transcript:
            asr_result["transcript"] = transcript
            annotation = annotate_language_spans(transcript, asr_result.get("language"))
            asr_result.update(annotation)
            asr_result["code_switch_detected"] = bool(
                annotation["code_switch_detected"] or annotation["mixed_script_detected"]
            )
            asr_result["romanized_hindi_detected"] = bool(annotation["romanized_hindi_detected"])
            if len(annotation["languages_present"]) == 1:
                asr_result["language"] = annotation["languages_present"][0]

        language = str(asr_result.get("language") or "unknown").lower().strip()
        quality_value = None if manual_transcript else asr_result.get("quality_score")
        if quality_value is None and not manual_transcript:
            quality_value = asr_result.get("confidence")
        try:
            asr_confidence = (
                max(0.0, min(1.0, float(quality_value)))
                if quality_value is not None else None
            )
        except (TypeError, ValueError):
            asr_confidence = None

        asr_quality_status = (
            "manual_review" if manual_transcript else
            ("unavailable" if asr_confidence is None else "uncalibrated_estimate")
        )
        if manual_transcript:
            asr_quality_note = "The transcript was manually corrected; the original ASR score does not apply."
        elif asr_confidence is None:
            asr_quality_note = (
                "The selected recognizer does not expose a usable transcript-quality "
                "score. Check the original transcript before relying on its English meaning."
            )
        else:
            asr_quality_note = (
                "The transcript-quality value is an uncalibrated model heuristic, "
                "not a probability that every recognised word is correct."
            )

        stage_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            text_future = (
                executor.submit(_run_timed, self._predict_text_emotion,
                                transcript, language, asr_result)
                if include_emotion else None
            )
            voice_future = None
            if include_emotion and "_voice_result" not in prepared:
                voice_future = executor.submit(_run_timed, self._predict_voice_emotion, audio)
            translation_future = (
                executor.submit(self._translate, transcript, asr_result)
                if include_translation else None
            )
            if text_future is not None:
                text_result, text_duration = text_future.result()
            else:
                text_result = _unavailable_branch("Skipped in fast case-intake mode.")
                text_duration = 0.0
            if not include_emotion:
                voice_result = _unavailable_branch("Skipped in fast case-intake mode.")
                voice_duration = 0.0
            elif voice_future is None:
                voice_result = prepared["_voice_result"]
                voice_duration = prepared.get("_voice_duration", 0.0)
            else:
                voice_result, voice_duration = voice_future.result()
                prepared["_voice_result"] = voice_result
                prepared["_voice_duration"] = voice_duration
            if translation_future is not None:
                translation_result = translation_future.result()
            else:
                translation_result = {
                    "text": "", "available": False, "source": None,
                    "source_language": language, "method": "skipped_fast_case_intake",
                    "note": "English translation was skipped to reduce case-intake latency. The original transcript remains available.",
                    "error": "Translation skipped in fast case-intake mode.",
                    "duration_sec": 0.0, "cache_used": None,
                    "translation_input_text": transcript,
                    "transliteration_available": None, "transliterated_text": None,
                    "transliteration_replacements": 0, "transliteration_model": None,
                    "transliteration_error": None,
                }
        timings["parallel_analysis_and_translation"] = round(
            time.perf_counter() - stage_start, 3
        )
        timings["text_emotion"] = text_duration
        timings["voice_emotion"] = voice_duration
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
        if manual_transcript:
            # Report work performed for this correction separately from work
            # already completed during the original recording analysis.
            original_stage_timings = {
                "preprocessing": timings.get("preprocessing", 0.0),
                "asr": timings.get("asr", 0.0),
                "voice_emotion": timings.get("voice_emotion", 0.0),
            }
            reused_stages = ["preprocessing", "asr"]
            timings["preprocessing"] = 0.0
            timings["asr"] = 0.0
            if "_voice_result" in prepared:
                reused_stages.append("voice_emotion")
                timings["voice_emotion"] = 0.0
            timings["original_stage_timings_sec"] = original_stage_timings
            timings["reused_stages"] = reused_stages

        timings["total"] = round(sum(timings.get(k, 0.0) for k in (
            "preprocessing", "asr", "parallel_analysis_and_translation", "fusion"
        )), 3)

        return {
            "transcript_source": "manual_correction" if manual_transcript else "automatic_asr",
            "audio_quality": audio_quality,
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
            "asr_candidate_initialization_seconds": asr_result.get(
                "asr_candidate_initialization_seconds"
            ),
            "asr_candidate_worker_request_seconds": asr_result.get(
                "asr_candidate_worker_request_seconds"
            ),
            "asr_context_terms": asr_result.get("asr_context_terms", ""),
            "asr_context_source": asr_result.get("asr_context_source", "none"),
            "asr_context_probe_seconds": asr_result.get(
                "asr_context_probe_seconds", 0.0
            ),
            "asr_context_probe_transcript": asr_result.get(
                "asr_context_probe_transcript", ""
            ),
            "asr_context_probe_language": asr_result.get(
                "asr_context_probe_language", ""
            ),
            "asr_context_probe_language_score": asr_result.get(
                "asr_context_probe_language_score"
            ),
            "asr_inference_device": asr_result.get("asr_inference_device"),
            "asr_worker_torch_threads": asr_result.get("asr_worker_torch_threads"),
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
            "translation_input_text": translation_result.get("translation_input_text"),
            "transliteration_available": translation_result.get("transliteration_available"),
            "transliterated_transcript": translation_result.get("transliterated_text"),
            "transliteration_replacements": translation_result.get("transliteration_replacements", 0),
            "transliteration_model": translation_result.get("transliteration_model"),
            "transliteration_error": translation_result.get("transliteration_error"),
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
            "text_emotion_decision_thresholds": text_result.get("decision_thresholds"),
            "text_emotion_active_labels": text_result.get("active_emotions"),
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
            "voice_top_two": voice_result.get("top_two", []),
            "voice_top_two_margin": voice_result.get("top_two_margin"),
            "voice_validation_macro_f1": voice_result.get("validation_macro_f1"),
            "text_emotion_inference_seconds": text_duration,
            "voice_emotion_inference_seconds": voice_duration,
            "torch_runtime": self.torch_runtime,
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
            "translator_initialization_seconds": (
                getattr(self.translator, "last_initialization_seconds", None)
                if self.translator is not None else None
            ),
            "translator_preparation_seconds": (
                getattr(self.translator, "last_preparation_seconds", None)
                if self.translator is not None else None
            ),
            "translator_generation_seconds": (
                getattr(self.translator, "last_generation_seconds", None)
                if self.translator is not None else None
            ),
            "translator_postprocessing_seconds": (
                getattr(self.translator, "last_postprocessing_seconds", None)
                if self.translator is not None else None
            ),
            "translator_worker_request_seconds": (
                getattr(self.translator, "last_worker_request_seconds", None)
                if self.translator is not None else None
            ),
            "translation_cache_used": translation_result.get("cache_used"),
            "translation_num_beams": (
                getattr(self.translator, "last_num_beams", None)
                if self.translator is not None else None
            ),
            "translator_error": translation_result["error"],
        }

    def run(
        self, audio_path: str, language_hint: str = None,
        asr_context: str = None, include_emotion: bool = True,
        include_translation: bool = True,
    ) -> dict:
        """Backward-compatible one-call API with opt-in fast intake switches."""
        prepared = self.prepare(audio_path, language_hint=language_hint,
                                asr_context=asr_context)
        return self.analyze_prepared(
            prepared, include_emotion=include_emotion,
            include_translation=include_translation,
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python pipeline.py <path_to_audio_file>")
    pipeline = SpeechMindPipeline()
    for key, value in pipeline.run(sys.argv[1]).items():
        print(f"{key:28s}: {value}")
