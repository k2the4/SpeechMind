import unittest

from models.asr_routing import asr_quality_requires_audio_fallback


class AsrQualityFallbackPolicyTests(unittest.TestCase):
    def test_only_valid_low_scores_trigger_audio_fallback(self):
        self.assertTrue(asr_quality_requires_audio_fallback(0.0))
        self.assertTrue(asr_quality_requires_audio_fallback(0.599))
        self.assertFalse(asr_quality_requires_audio_fallback(0.60))
        self.assertFalse(asr_quality_requires_audio_fallback(0.95))

    def test_missing_or_malformed_scores_are_unknown_not_low(self):
        for value in (None, "", "unknown", object()):
            with self.subTest(value=value):
                self.assertFalse(asr_quality_requires_audio_fallback(value))

    def test_non_finite_and_out_of_range_scores_are_rejected(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value):
                self.assertFalse(asr_quality_requires_audio_fallback(value))


if __name__ == "__main__":
    unittest.main()
