"""
SpeechMind end-to-end inference pipeline.

ASR and translation providers are configurable. Hosted Bodhan providers are
opt-in because they transmit audio/transcript data externally and incur usage
charges. The default remains the existing local provider.
"""
import os
import time
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
        self.translator_provider = os.getenv(
            "SPEECHMIND_TRANSLATOR_PROVIDER", "local"
        ).strip().lower()
        if self.translator_provider not in {"local", "bodhan"}:
            raise ValueError(
                "SPEECHMIND_TRANSLATOR_PROVIDER must be either 'local' or 'bodhan'."
            )
        self.translator_error = None
        print(
            "[Pipeline] Core models initialized. "
            f"ASR provider={os.getenv('SPEECHMIND_ASR_PROVIDER', 'local')}; "
            f"translation provider={self.translator_provider}."
        )

    def _translate(self, transcript: str, language: str) -> str:
        if not transcript:
            return transcript
        if not self.translator_enabled:
            return transcript

        # Bodhan Indic-Translate accepts code-mixed input without requiring a
        # single source-language code. Call it even if Whisper's primary
        # language guess is English, because the utterance may still be mixed.
        if self.translator_provider == "bodhan":
            try:
                if self.translator is None:
                    print("[Pipeline] Loading Bodhan Indic-Translate API adapter...")
                    from models.bodhan_translation import BodhanTranslator
                    self.translator = BodhanTranslator()
                return self.translator.to_english(transcript, language)
            except Exception as error:
                self.translator_error = str(error)
                print(f"[Pipeline] Bodhan translation unavailable: {error}")
                return transcript

        # Existing local IndicTrans2 path remains available as the default.
        if language == "en" or language not in TRANSLATABLE_LANGUAGES:
            return transcript
        try:
            if self.translator is None:
                print("[Pipeline] Loading IndicTrans2 on demand...")
                from models.translation import Translator
                self.translator = Translator()
            return self.translator.to_english(transcript, language)
        except Exception as error:
            self.translator_error = str(error)
            print(f"[Pipeline] Translation unavailable: {error}")
            return transcript

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
            asr_confidence=asr_confidence,
        )
        timings["fusion"] = round(time.perf_counter() - stage_start, 3)

        stage_start = time.perf_counter()
        english_meaning = self._translate(transcript, language)
        timings["translation"] = round(time.perf_counter() - stage_start, 3)
        timings["total"] = round(time.perf_counter() - total_start, 3)

        return {
            "language": language,
            "language_detection_score": asr_result.get("language_detection_score"),
            "language_source": asr_result.get("language_source"),
            "transcript": transcript,
            "asr_model": asr_result.get("asr_model", "unknown"),
            "asr_provider": asr_result.get("provider", "local"),
            "asr_fallback": bool(asr_result.get("asr_fallback", False)),
            "asr_fallback_reason": asr_result.get("asr_fallback_reason"),
            "asr_requested_model": asr_result.get("asr_requested_model"),
            "asr_confidence": round(asr_confidence, 3) if asr_confidence is not None else None,
            "asr_quality_score": (
                round(asr_confidence, 3) if asr_confidence is not None else None
            ),
            "asr_quality_method": asr_result.get("quality_method"),
            "asr_avg_logprob": asr_result.get("avg_logprob"),
            "asr_avg_no_speech_prob": asr_result.get("avg_no_speech_prob"),
            "asr_avg_compression_ratio": asr_result.get("avg_compression_ratio"),
            "meaning_en": english_meaning,
            "text_emotion": fused["text_emotion"],
            "text_emotion_available": bool(text_result.get("available", True)),
            "text_emotion_reason": text_result.get("reason"),
            "text_emotion_model_source": text_result.get("model_source"),
            "text_emotion_languages": sorted(self.text_emotion.supported_languages),
            "voice_emotion": fused["voice_emotion"],
            "voice_model_source": voice_result.get("model_source"),
            "voice_inference_device": voice_result.get("inference_device"),
            "voice_inference_windows": voice_result.get("inference_windows"),
            "voice_validation_macro_f1": voice_result.get("validation_macro_f1"),
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
            "translator_provider": self.translator_provider,
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
