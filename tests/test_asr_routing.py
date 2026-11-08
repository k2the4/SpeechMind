import unittest
from unittest.mock import Mock, patch

import numpy as np

from models.asr import SpeechRecognizer


class ASRRoutingTests(unittest.TestCase):
    def test_models_are_not_loaded_during_router_initialization(self):
        recognizer = SpeechRecognizer()
        self.assertIsNone(recognizer.whisper)
        self.assertIsNone(recognizer.indic_model)

    def test_selected_hindi_routes_directly_to_indic_without_whisper(self):
        recognizer = SpeechRecognizer()
        with patch.object(recognizer, "_indic_transcribe", return_value="नमस्ते") as indic:
            result = recognizer.transcribe(
                np.zeros(16000, dtype=np.float32),
                language_hint="hi",
            )
        indic.assert_called_once()
        self.assertIsNone(recognizer.whisper)
        self.assertEqual(result["language"], "hi")
        self.assertEqual(result["language_route"], "selected-indic")

    def test_selected_english_routes_directly_to_whisper_without_indic(self):
        recognizer = SpeechRecognizer()
        fake_whisper = Mock()
        fake_whisper.transcribe.return_value = {
            "text": "Hello there",
            "segments": [{"avg_logprob": -0.2}],
        }
        with patch.object(recognizer, "_load_whisper", return_value=fake_whisper):
            result = recognizer.transcribe(
                np.zeros(16000, dtype=np.float32),
                language_hint="en",
            )
        fake_whisper.transcribe.assert_called_once()
        self.assertIsNone(recognizer.indic_model)
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["language_route"], "selected-english")

    def test_invalid_language_hint_is_rejected(self):
        recognizer = SpeechRecognizer()
        with self.assertRaises(ValueError):
            recognizer.transcribe(
                np.zeros(16000, dtype=np.float32),
                language_hint="xx",
            )


if __name__ == "__main__":
    unittest.main()
