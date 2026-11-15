import unittest
from unittest.mock import patch

import numpy as np

from utils.audio_utils import prepare_audio_paths


class AudioPreparationTests(unittest.TestCase):
    @patch("utils.audio_utils.analyze_audio_quality")
    @patch("utils.audio_utils.load_audio")
    def test_asr_gets_dc_corrected_copy_while_voice_keeps_original(
        self, mocked_load, mocked_quality
    ):
        raw = np.asarray([0.35, 0.45, 0.25, 0.55], dtype=np.float32)
        quality = {"status": "good"}
        mocked_load.return_value = raw.copy()
        mocked_quality.return_value = quality

        voice_audio, asr_audio, returned_quality = prepare_audio_paths("sample.wav")

        expected_voice = raw / np.max(np.abs(raw))
        np.testing.assert_allclose(voice_audio, expected_voice, atol=1e-6)
        self.assertAlmostEqual(float(np.mean(asr_audio)), 0.0, places=6)
        self.assertFalse(np.allclose(voice_audio, asr_audio))
        self.assertIsNot(voice_audio, asr_audio)
        self.assertIs(returned_quality, quality)
        mocked_quality.assert_called_once()
        np.testing.assert_array_equal(mocked_quality.call_args.args[0], raw)

    @patch("utils.audio_utils.analyze_audio_quality")
    @patch("utils.audio_utils.load_audio")
    def test_silent_audio_remains_finite_for_both_branches(
        self, mocked_load, mocked_quality
    ):
        mocked_load.return_value = np.zeros(8, dtype=np.float32)
        mocked_quality.return_value = {"status": "unusable"}

        voice_audio, asr_audio, _ = prepare_audio_paths("silent.wav")

        self.assertTrue(np.isfinite(voice_audio).all())
        self.assertTrue(np.isfinite(asr_audio).all())
        np.testing.assert_array_equal(voice_audio, np.zeros(8, dtype=np.float32))
        np.testing.assert_array_equal(asr_audio, np.zeros(8, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
