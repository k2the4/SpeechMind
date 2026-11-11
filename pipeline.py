"""
SpeechMind end-to-end local inference pipeline.

The original transcript, English meaning, text analysis, voice analysis, and
combined interpretation remain distinct outputs.
"""
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from utils.audio_utils import preprocess
from models.asr import SpeechRecognizer
from models.text_emotion import TextEmotionClassifier
from models.voice_emotion import VoiceEmotionRecognizer
from fusion_layer import fuse

TRANSLATABLE_LANGUAGES = {
    "hi", "bn", "ta", "te", "mr", "gu", "pa", "ml", "kn", "ur"
}


class SpeechMindPipeline:
    def __init__(self, load_translator: bool = True):
        print("[Pipeline] Initializing SpeechMind models...")
        self.asr = SpeechRecognizer()
        self.text_emotion = TextEmotionClassifier()
        self.voice_emotion = VoiceEmotionRecognizer()
        self.translator = None
        self.translator_enabled = load_translator
        self._translator_lock = threading.Lock()
        print(
            "[Pipeline] Core local models initialized. "
            "No hosted or metered inference service is configured."
        )

    def _get_translator(self):
        # Streamlit can serve repeated requests through one cached pipeline.
        # Initialize the isolated worker once even if two requests arrive together.
        with self._translator_lock:
            if self.translator is None:
                print("[Pipeline] Initializing isolated IndicTrans2 translator...")
                from models.translation import Translator
                self.translator = Translator()
            return self.translator

    def _translate(self, transcript: str, language: str) -> dict:
        started = time.perf_counter()
        result = {
            "text": transcript,
            "available": False,
            "source": None,
            "error": None,
            "duration_sec": 0.0,
        }
        try:
            if not transcript:
                result["error"] = "No transcript was produced."
            elif language in ("en", "english"):
                result.update(
                    text=transcript,
                    available=True,
                    source="original_english_transcript",
                )
            elif not self.translator_enabled:
                result["error"] = "Translation is disabled."
            elif language not in TRANSLATABLE_LANGUAGES:
                result["error"] = (
                    f"English meaning is not configured for detected language: {language}."
                )
            else:
                translator = self._get_translator()
                translated = translator.to_english(transcript, language)
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

    def run(self, audio_path: str, language_hint: str = None) -> dict:
        total_start = time.perf_counter()
        timings = {}

        stage_start = time.perf_counter()
        audio = preprocess(audio_path)
        timings["preprocessing"] = round(time.perf_counter() - stage_start, 3)

        stage_start = time.perf_counter()
        asr_result = self.asr.transcribe(audio, language_hint=language_hint)
        timings["asr"] = round(time.perf_counter() - stage_start, 3)

        transcript = asr_result.get("transcript", "")
        language = asr_result.get("language", "unknown")
        quality_value = asr_result.get("quality_score", asr_result.get("confidence"))
        try:
            asr_confidence = (
                max(0.0, min(1.0, float(quality_value)))
                if quality_value is not None else None
            )
        except (TypeError, ValueError):
            asr_confidence = None

        stage_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            text_future = executor.submit(
                self.text_emotion.predict, transcript, language
            )
            voice_future = executor.submit(self.voice_emotion.predict, audio)
            translation_future = executor.submit(
                self._translate, transcript, language
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
        timings["total"] = round(time.perf_counter() - total_start, 3)

        return {
            "language": language,
            "language_detection_score": asr_result.get("language_detection_score"),
            "language_source": asr_result.get("language_source"),
            "transcript": transcript,
            "asr_model": asr_result.get("asr_model", "unknown"),
            "asr_provider": "local",
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
            "asr_confidence": round(asr_confidence, 3) if asr_confidence is not None else None,
            "asr_quality_score": (
                round(asr_confidence, 3) if asr_confidence is not None else None
            ),
            "asr_quality_method": asr_result.get("quality_method"),
            "asr_avg_logprob": asr_result.get("avg_logprob"),
            "asr_avg_no_speech_prob": asr_result.get("avg_no_speech_prob"),
            "asr_avg_compression_ratio": asr_result.get("avg_compression_ratio"),
            "meaning_en": english_meaning,
            "meaning_en_available": translation_result["available"],
            "meaning_en_source": translation_result["source"],
            "text_emotion": fused["text_emotion"],
            "text_emotion_available": bool(text_result.get("available", True)),
            "text_emotion_reason": text_result.get("reason"),
            "text_emotion_model_source": text_result.get("model_source"),
            "text_emotion_secondary": text_result.get("secondary_emotions", []),
            "text_emotion_multi_label_scores": text_result.get("multi_label_scores"),
            "text_emotion_task_type": text_result.get("task_type"),
            "text_emotion_languages": sorted(self.text_emotion.supported_languages),
            "voice_emotion": fused["voice_emotion"],
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
            "branch_comparison": fused["branch_comparison"],
            "interpretation_status": fused["interpretation_status"],
            "uncertainty_reasons": fused["uncertainty_reasons"],
            "text_emotion_distribution": fused["text_distribution"],
            "voice_emotion_distribution": fused["voice_distribution"],
            "fused_emotion_distribution": fused["fused_distribution"],
            "processing_time_sec": timings["total"],
            "timings_sec": timings,
            "translator_provider": (
                self.translator.provider_name if self.translator is not None
                else ("disabled" if not self.translator_enabled else "not_initialized")
            ),
            "translator_loaded": bool(
                self.translator is not None and self.translator.loaded
            ),
            "translator_inference_seconds": (
                self.translator.last_inference_seconds
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
