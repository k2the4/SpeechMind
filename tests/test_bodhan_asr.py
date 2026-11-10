import unittest
from unittest.mock import Mock, patch

import numpy as np

from models.bodhan_asr import BodhanASR


class BodhanASRTests(unittest.TestCase):
    def setUp(self):
        self.client = BodhanASR(api_key="test-key")

    @patch("models.bodhan_asr.requests.post")
    def test_transcribes_audio_without_forcing_a_language(self, post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"text": "Kal I have an exam."}
        post.return_value = response

        result = self.client.transcribe(np.zeros(16000, dtype=np.float32))

        self.assertEqual(result["transcript"], "Kal I have an exam.")
        self.assertEqual(result["asr_model"], "bodhan-indic-transcribe-api")
        self.assertIsNone(result["quality_score"])
        self.assertEqual(result["language_source"], "provider_auto_code_mixed")
        self.assertEqual(post.call_args.kwargs["data"], {"model": "indic-transcribe"})
        self.assertNotIn("language", post.call_args.kwargs["data"])
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key"
        )

    @patch("models.bodhan_asr.requests.post")
    def test_rejects_audio_longer_than_provider_limit(self, post):
        audio = np.zeros(31 * 16000, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "up to 30 seconds"):
            self.client.transcribe(audio)
        post.assert_not_called()

    @patch("models.bodhan_asr.requests.post")
    def test_rejects_empty_transcript(self, post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"text": "   "}
        post.return_value = response
        with self.assertRaisesRegex(RuntimeError, "empty transcript"):
            self.client.transcribe(np.zeros(16000, dtype=np.float32))

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "BODHAN_API_KEY"):
                BodhanASR()


if __name__ == "__main__":
    unittest.main()
