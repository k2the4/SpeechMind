"""
SpeechMind end-to-end inference pipeline.

The user never specifies a language. The ASR backend performs automatic
language handling; the English meaning branch accepts the original code-mixed
transcript directly when the Bodhan API is configured.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from utils.audio_utils import preprocess
from models.asr import SpeechRecognizer
from models.bodhan_client import BodhanClient
from models.text_emotion import TextEmotionClassifier
from models.voice_emotion import VoiceEmotionRecognizer
from models.indic_language_id import IndicLanguageIdentifier
from fusion_layer import fuse

TRANSLATABLE_LANGUAGES = {
    "hi", "bn", "ta", "te", "mr", "gu", "pa", "ml", "kn", "ur"
}


class SpeechMindPipeline:
    def __init__(self, load_translator: bool = True):
        print("[Pipeline] Initializing SpeechMind models...")
        self.bodhan_client = BodhanClient.from_env()
        self.asr = SpeechRecognizer(
            backend="auto", bodhan_client=self.bodhan_client
        )
        self.language_identifier = IndicLanguageIdentifier()
        self.text_emotion = TextEmotionClassifier()
        self.voice_emotion = VoiceEmotionRecognizer()
        self.translator = None
        self.translator_enabled = load_translator
        self.translator_error = None
        self.translation_provider = "none"
        print(
            "[Pipeline] ASR backend: "
            + self.asr.provider
            + ". Text and voice emotion branches are initialized."
        )

    def _translate(self, transcript: str, language: str) -> str:
        if not transcript or not transcript.strip():
            self.translation_provider = "none"
            return transcript
        if not self.translator_enabled:
            self.translation_provider = "disabled"
            return transcript

        # The Bodhan translation endpoint infers the source and accepts Romanized
        # and code-mixed Indic text; do not pass a guessed source-language code.
        if self.bodhan_client is not None:
            try:
                result = self.bodhan_client.translate_to_english(transcript)
                self.translation_provider = result.get(
                    "provider", "bodhan_indic_translate"
                )
                self.translator_error = None
                return result["translation"]
            except Exception as error:
                self.translator_error = str(error)
                print(f"[Pipeline] Bodhan translation unavailable: {error}")

        # Offline fallback is safe only when the transcript has one known Indic
        # language. Never feed a guessed language code for a code-mixed transcript.
        if language not in TRANSLATABLE_LANGUAGES:
            self.translation_provider = "unavailable_offline"
            return transcript
        try:
            if self.translator is None:
                print("[Pipeline] Loading offline IndicTrans2 on demand...")
                from models.translation import Translator
                self.translator = Translator()
            translated = self.translator.to_english(transcript, language)
            self.translation_provider = "local_indictrans2"
            return translated
        except Exception as error:
            self.translator_error = str(error)
            self.translation_provider = "unavailable_offline"
            print(f"[Pipeline] Offline translation unavailable: {error}")
            return transcript

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
        language_id_result = self.language_identifier.predict(transcript)
        # Prefer segment-level IndicLID when its weights are installed. The
        # script-only fallback is explicit and is not reported as model LID.
        asr_result.update(language_id_result)
        language = asr_result.get(
            "language_profile", asr_result.get("language", "unknown")
        )
        quality_value = asr_result.get(
            "quality_score", asr_result.get("confidence")
        )
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
                self.text_emotion.predict,
                transcript,
                asr_result.get("language_profile", language),
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
            "language_profile": asr_result.get(
                "language_profile", language
            ),
            "language_profile_note": asr_result.get("language_profile_note"),
            "language_identifier": asr_result.get("language_identifier"),
            "language_id_available": asr_result.get("language_id_available", False),
            "language_segments": asr_result.get("language_segments", []),
            "script_evidence": asr_result.get("script_evidence", []),
            "language_detection_score": asr_result.get("language_detection_score"),
            "language_source": asr_result.get("language_source"),
            "transcript": transcript,
            "asr_model": asr_result.get("asr_model", "unknown"),
            "asr_provider": asr_result.get("asr_provider", self.asr.provider),
            "asr_provider_fallback_reason": asr_result.get(
                "asr_provider_fallback_reason"
            ),
            "asr_chunks": asr_result.get("asr_chunks", 1),
            "asr_confidence": (
                round(asr_quality, 3) if asr_quality is not None else None
            ),
            "asr_quality_score": (
                round(asr_quality, 3) if asr_quality is not None else None
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
            "text_emotion_languages": sorted(
                getattr(self.text_emotion, "supported_languages", [])
            ),
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
            "text_distribution": fused.get("text_distribution"),
            "voice_distribution": fused.get("voice_distribution"),
            "fused_distribution": fused.get("fused_distribution"),
            "processing_time_sec": timings["total"],
            "timings_sec": timings,
            "translator_loaded": self.translator is not None,
            "translation_provider": self.translation_provider,
            "translator_error": self.translator_error,
            "cloud_processing_enabled": self.bodhan_client is not None,
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python pipeline.py <path_to_audio_file>")
    pipeline = SpeechMindPipeline()
    for key, value in pipeline.run(sys.argv[1]).items():
        print(f"{key:28s}: {value}")
