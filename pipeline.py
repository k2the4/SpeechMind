"""
SpeechMind end-to-end inference pipeline.

ASR routing is lazy: a selected language loads only its corresponding ASR
model. Auto-detect retains the original language-identification flow.
IndicTrans2 is loaded only for a supported non-English transcript.
"""
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
        print("[Pipeline] Initializing SpeechMind...")
        self.asr = SpeechRecognizer()
        self.text_emotion = TextEmotionClassifier()
        self.voice_emotion = VoiceEmotionRecognizer()
        self.translator = None
        self.translator_enabled = load_translator
        self.translator_error = None
        print("[Pipeline] Core classifier ready; ASR and translation weights are lazy-loaded.")

    def _translate(self, transcript: str, language: str) -> str:
        if not transcript or language == "en":
            return transcript
        if not self.translator_enabled or language not in TRANSLATABLE_LANGUAGES:
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

    def run(self, audio_path: str, language_hint=None) -> dict:
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
        try:
            asr_confidence = float(asr_result.get("confidence", 0.0))
        except (TypeError, ValueError):
            asr_confidence = 0.0
        asr_confidence = max(0.0, min(1.0, asr_confidence))

        stage_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(self.text_emotion.predict, transcript)
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
            "language_route": asr_result.get("language_route", "auto"),
            "asr_model": asr_result.get("asr_model", "unknown"),
            "transcript": transcript,
            "asr_confidence": round(asr_confidence, 3),
            "asr_confidence_method": asr_result.get("confidence_method"),
            "meaning_en": english_meaning,
            "text_emotion": fused["text_emotion"],
            "text_emotion_available": bool(text_result.get("available", True)),
            "text_emotion_reason": text_result.get("reason"),
            "voice_emotion": fused["voice_emotion"],
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
