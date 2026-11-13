import unittest
import numpy as np
from utils.audio_quality import analyze_audio_quality


class AudioQualityTests(unittest.TestCase):
    def test_clean_signal(self):
        audio = (np.sin(np.linspace(0, 10, 16000, endpoint=False)) * 0.2).astype(np.float32)
        result = analyze_audio_quality(audio)
        self.assertEqual(result["duration_seconds"], 1.0)
        self.assertEqual(result["status"], "good")
        self.assertFalse(result["is_probability"])

    def test_clipping(self):
        audio = np.zeros(16000, dtype=np.float32)
        audio[:100] = 1.0
        result = analyze_audio_quality(audio)
        self.assertIn("clipping_detected", result["flags"])
        self.assertEqual(result["status"], "review")

    def test_silence(self):
        result = analyze_audio_quality(np.zeros(8000, dtype=np.float32))
        self.assertEqual(result["status"], "unusable")
        self.assertIn("near_silent", result["flags"])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            analyze_audio_quality(np.array([], dtype=np.float32))
        with self.assertRaises(ValueError):
            analyze_audio_quality(np.array([np.nan], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
