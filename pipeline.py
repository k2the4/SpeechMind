"""
SpeechMind end-to-end automatic multilingual speech-understanding pipeline.

The user does not select a language. ASR transcribes automatically; translation
creates a separate English meaning; text and voice emotion are analyzed in
parallel; fusion preserves agreement/disagreement and uncertainty.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from utils.audio_utils import preprocess
from models.asr import SpeechRecognizer
from models.text_emotion import TextEmotionClassifier
from models.voice_emotion import VoiceEmotionRecognizer
from fusion_layer import fuse


class SpeechMindPipeline:
    def __init__(self, load_translator: bool = True):
        print("[Pipeline] Initializing SpeechMind models...")
        self.asr = SpeechRecognizer()
        self.text_emotion = TextEmotionClassifier()
        self.voice_emotion = VoiceEmotionRecognizer()
        self.translator = None
        self.translator_enabled = load_translator
        self.translator_error = None
        self.translation_backend = None
        print("[Pipeline] Core models initialized. Optional translation remains lazy-loaded.")

    def _translate(self, transcript: str, language: str):
        if not transcript or not self.translator_enabled:
            return None
        try:
            if self.translator is None:
                from models.translation import Translator
                self.translator = Translator()
            result = self.translator.to_english(transcript, language)
            self.translation_backend = self.translator.backend
            self.translator_error = self.translator.last_reason
            return result
        except Exception as error:
            self.translator_error = str(error)
            print(f"[Pipeline] Translation unavailable: {error}")
            return None

    def run(self, audio_path: str) -> dict:
        total_start = time.perf_counter()
        timings = {}

        stage_start = time.perf_counter()
        audio = preprocess(audio_path)
        timings["preprocessing"] = round(time.perf_counter() - stage_start, 3)

        stage_start = time.perf_counter()
        asr_result = self.asr.transcribe(audio)
        timings["asr"] = round(time.perf_counter() - stage_start, 3)

        transcript = asr_result.get("transcript", "")
        language = asr_result.get("language", "unknown")
        quality_value = asr_result.get("quality_score", asr_result.get("confidence"))
        try:
            asr_quality = (
                max(0.0, min(1.0, float(quality_value)))
                if quality_value is not None else None
            )
        except (TypeError, ValueError):
            asr_quality = None

        stage_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(
                self.text_emotion.predict, transcript, language
            )
            voice_future = executor.submit(self.voice_emotion.predict, audio)
            text_result = text_future.result()
            voice_result = voice_future.result()
        timings["emotion_branches"] = round(time.perf_counter() - stage_start, 3)

        stage_start = time.perf_counter()
        fused = fuse(
            text_result=text_result,
            voice_result=voice_result,
            asr_confidence=asr_quality,
        )
        timings["fusion"] = round(time.perf_counter() - stage_start, 3)

        stage_start = time.perf_counter()
        english_meaning = self._translate(transcript, language)
        timings["translation"] = round(time.perf_counter() - stage_start, 3)
        timings["total"] = round(time.perf_counter() - total_start, 3)

        return {
            "language": language,
            "languages_detected": asr_result.get("languages_detected", []),
            "language_detection_score": asr_result.get("language_detection_score"),
            "language_source": asr_result.get("language_source"),
            "transcript": transcript,
            "asr_model": asr_result.get("asr_model", "unknown"),
            "asr_backend": asr_result.get("asr_backend", "unknown"),
            "asr_confidence": round(asr_quality, 3) if asr_quality is not None else None,
            "asr_quality_score": round(asr_quality, 3) if asr_quality is not None else None,
            "asr_quality_method": asr_result.get("quality_method"),
            "asr_avg_logprob": asr_result.get("avg_logprob"),
            "asr_avg_no_speech_prob": asr_result.get("avg_no_speech_prob"),
            "asr_avg_compression_ratio": asr_result.get("avg_compression_ratio"),
            "asr_chunk_count": asr_result.get("asr_chunk_count", 1),
            "asr_segments": asr_result.get("asr_segments", []),
            "meaning_en": english_meaning,
            "translation_available": bool(english_meaning),
            "translation_backend": self.translation_backend,
            "translation_reason": self.translator_error,
            "text_emotion": fused["text_emotion"],
            "text_emotion_available": bool(text_result.get("available", True)),
            "text_emotion_reason": text_result.get("reason"),
            "text_emotion_model_source": text_result.get("model_source"),
            "text_emotion_languages": sorted(self.text_emotion.supported_languages),
            "text_distribution": fused.get("text_distribution", {}),
            "voice_emotion": fused["voice_emotion"],
            "voice_model_source": voice_result.get("model_source"),
            "voice_inference_device": voice_result.get("inference_device"),
            "voice_inference_windows": voice_result.get("inference_windows"),
            "voice_validation_macro_f1": voice_result.get("validation_macro_f1"),
            "voice_distribution": fused.get("voice_distribution", {}),
            "signals_agree": fused.get("signals_agree"),
            "signal_agreement": fused.get("signal_agreement"),
            "valence": fused["valence"],
            "arousal": fused["arousal"],
            "overall_interpretation": fused["overall_interpretation"],
            "confidence": fused["confidence"],
            "confidence_is_calibrated": False,
            "low_confidence_warning": fused["low_confidence_warning"],
            "overall_emotion": fused["overall_emotion"],
            "fusion_weights": fused["fusion_weights"],
            "processing_time_sec": timings["total"],
            "timings_sec": timings,
            "translator_loaded": self.translator is not None,
            "translator_error": self.translator_error,
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python pipeline.py <path_to_audio_file>")
    pipeline = SpeechMindPipeline()
    for key, value in pipeline.run(sys.argv[1]).items():
        print(f"{key:28s}: {value}")
