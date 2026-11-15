import importlib
import sys
import types
import unittest
from unittest.mock import patch


# The lightweight CI job intentionally omits librosa, torch, and Transformers.
# Stub only pipeline dependencies at import time; each test replaces the model
# factories with deterministic fakes and exercises the real orchestration code.
_audio_module = types.ModuleType("utils.audio_utils")
_audio_module.preprocess = lambda _path: [0.1, 0.2, 0.1]
_audio_module.preprocess_with_quality = lambda _path: (
    [0.1, 0.2, 0.1], {"status": "good", "duration_seconds": 1.0, "flags": []}
)
_audio_module.prepare_audio_paths = lambda _path: (
    [0.1, 0.2, 0.1],
    [0.1, 0.2, 0.1],
    {"status": "good", "duration_seconds": 1.0, "flags": []},
)
_asr_module = types.ModuleType("models.asr")
_asr_module.SpeechRecognizer = object
_text_module = types.ModuleType("models.text_emotion")
_text_module.TextEmotionClassifier = object
_voice_module = types.ModuleType("models.voice_emotion")
_voice_module.VoiceEmotionRecognizer = object

with patch.dict(sys.modules, {
    "utils.audio_utils": _audio_module,
    "models.asr": _asr_module,
    "models.text_emotion": _text_module,
    "models.voice_emotion": _voice_module,
}):
    pipeline_module = importlib.import_module("pipeline")


class FakeASR:
    def transcribe(self, audio, language_hint=None):
        return {
            "transcript": "I am happy today",
            "language": "en",
            "quality_score": 0.9,
            "asr_model": "fake-asr",
        }


class FakeTextEmotion:
    supported_languages = {"en"}

    def predict(self, transcript, language):
        return {
            "emotion": "joy",
            "confidence": 0.9,
            "distribution": {"joy": 0.9, "neutral": 0.1},
            "available": True,
            "reason": None,
        }


class FakeVoiceEmotion:
    def predict(self, audio):
        return {
            "emotion": "sadness",
            "confidence": 0.8,
            "distribution": {"sadness": 0.8, "neutral": 0.2},
            "available": True,
            "model_source": "fake-voice",
        }


class PipelineFailureIsolationTests(unittest.TestCase):
    def _run_with_patched_components(
        self, asr_factory=FakeASR, text_factory=FakeTextEmotion,
        voice_factory=FakeVoiceEmotion
    ):
        with (
            patch.object(pipeline_module, "SpeechRecognizer", asr_factory),
            patch.object(pipeline_module, "TextEmotionClassifier", text_factory),
            patch.object(pipeline_module, "VoiceEmotionRecognizer", voice_factory),
            patch.object(
                pipeline_module, "preprocess_with_quality",
                return_value=(
                    [0.1, 0.2, 0.1],
                    {"status": "good", "duration_seconds": 1.0, "flags": []},
                ),
                create=True,
            ),
            patch.object(
                pipeline_module, "prepare_audio_paths",
                return_value=(
                    [0.1, 0.2, 0.1],
                    [0.1, 0.2, 0.1],
                    {"status": "good", "duration_seconds": 1.0, "flags": []},
                ),
                create=True,
            ),
        ):
            pipeline = pipeline_module.SpeechMindPipeline(load_translator=False)
            return pipeline.run("unused.wav")

    def test_voice_branch_survives_asr_initialization_failure(self):
        def unavailable_asr():
            raise RuntimeError("ASR weights unavailable")

        result = self._run_with_patched_components(asr_factory=unavailable_asr)

        self.assertIn("ASR weights unavailable", result["asr_error"])
        self.assertEqual(result["transcript"], "")
        self.assertFalse(result["text_branch_available"])
        self.assertTrue(result["voice_branch_available"])
        self.assertEqual(result["interpretation_status"], "single_branch")
        self.assertEqual(result["overall_emotion"], "sadness")

    def test_text_branch_failure_does_not_block_voice_or_english_meaning(self):
        class BrokenTextEmotion:
            supported_languages = {"en"}

            def predict(self, transcript, language):
                raise RuntimeError("text model failed")

        result = self._run_with_patched_components(text_factory=BrokenTextEmotion)

        self.assertEqual(result["transcript"], "I am happy today")
        self.assertEqual(result["meaning_en"], "I am happy today")
        self.assertTrue(result["meaning_en_available"])
        self.assertFalse(result["text_branch_available"])
        self.assertTrue(result["voice_branch_available"])
        self.assertIn("text model failed", result["text_emotion_reason"])
        self.assertEqual(result["interpretation_status"], "single_branch")

    def test_voice_branch_failure_does_not_block_text_analysis(self):
        class BrokenVoiceEmotion:
            def predict(self, audio):
                raise RuntimeError("voice model failed")

        result = self._run_with_patched_components(voice_factory=BrokenVoiceEmotion)

        self.assertTrue(result["text_branch_available"])
        self.assertFalse(result["voice_branch_available"])
        self.assertIn("voice model failed", result["voice_emotion_reason"])
        self.assertEqual(result["interpretation_status"], "single_branch")
        self.assertEqual(result["overall_emotion"], "joy")

    def test_pipeline_reports_independent_model_branch_timings(self):
        result = self._run_with_patched_components()

        self.assertIn("text_emotion", result["timings_sec"])
        self.assertIn("voice_emotion", result["timings_sec"])
        self.assertIn("translation", result["timings_sec"])
        self.assertGreaterEqual(result["text_emotion_inference_seconds"], 0.0)
        self.assertGreaterEqual(result["voice_emotion_inference_seconds"], 0.0)
        self.assertIn("torch_runtime", result)
        self.assertIsInstance(result["torch_runtime"]["available"], bool)

    def test_no_emotion_branches_returns_insufficient_evidence_not_neutral(self):
        def unavailable_component():
            raise RuntimeError("component unavailable")

        result = self._run_with_patched_components(
            asr_factory=unavailable_component,
            text_factory=unavailable_component,
            voice_factory=unavailable_component,
        )

        self.assertEqual(result["interpretation_status"], "insufficient_evidence")
        self.assertEqual(result["overall_emotion"], "unavailable")
        self.assertEqual(result["text_emotion"], "unavailable")
        self.assertEqual(result["voice_emotion"], "unavailable")
        self.assertEqual(result["fusion_weights"], {"text": 0.0, "voice": 0.0})
        self.assertIsNone(result["confidence"])
        self.assertFalse(result["text_branch_available"])
        self.assertFalse(result["voice_branch_available"])


    def test_transcript_correction_does_not_repeat_asr_or_voice(self):
        class CountingASR:
            def __init__(self): self.calls = 0
            def transcribe(self, audio, language_hint=None):
                self.calls += 1
                return {"transcript": "I am happy today", "language": "en",
                        "quality_score": 0.9, "asr_model": "fake-asr"}

        asr = CountingASR()
        voice_calls, text_calls = [], []
        class Text(FakeTextEmotion):
            def predict(self, transcript, language):
                text_calls.append(transcript)
                return super().predict(transcript, language)
        class Voice(FakeVoiceEmotion):
            def predict(self, audio):
                voice_calls.append(True)
                return super().predict(audio)

        with (
            patch.object(pipeline_module, "SpeechRecognizer", return_value=asr),
            patch.object(pipeline_module, "TextEmotionClassifier", Text),
            patch.object(pipeline_module, "VoiceEmotionRecognizer", Voice),
            patch.object(pipeline_module, "preprocess_with_quality",
                         return_value=([0.1, 0.2, 0.1],
                         {"status": "good", "duration_seconds": 1.0, "flags": []}),
                         create=True),
            patch.object(pipeline_module, "prepare_audio_paths",
                         return_value=([0.1, 0.2, 0.1], [0.1, 0.2, 0.1],
                         {"status": "good", "duration_seconds": 1.0, "flags": []}),
                         create=True),
        ):
            pipeline = pipeline_module.SpeechMindPipeline(load_translator=False)
            prepared = pipeline.prepare("unused.wav")
            first = pipeline.analyze_prepared(prepared)
            corrected = pipeline.analyze_prepared(
                prepared, transcript_override="I am genuinely happy today"
            )
        self.assertEqual(asr.calls, 1)
        self.assertEqual(len(voice_calls), 1)
        self.assertEqual(first["transcript_source"], "automatic_asr")
        self.assertEqual(corrected["transcript_source"], "manual_correction")
        self.assertEqual(corrected["transcript"], "I am genuinely happy today")
        self.assertEqual(corrected["asr_quality_status"], "manual_review")
        self.assertIsNone(corrected["asr_quality_score"])
        self.assertEqual(text_calls[-1], "I am genuinely happy today")
        self.assertEqual(corrected["timings_sec"]["asr"], 0.0)
        self.assertEqual(corrected["timings_sec"]["preprocessing"], 0.0)
        self.assertIn("asr", corrected["timings_sec"]["reused_stages"])
        self.assertIn("voice_emotion", corrected["timings_sec"]["reused_stages"])
        self.assertIn("original_stage_timings_sec", corrected["timings_sec"])
        self.assertGreaterEqual(corrected["timings_sec"]["original_stage_timings_sec"]["asr"], 0.0)


if __name__ == "__main__":
    unittest.main()
