import math
import unittest

from models.asr_quality import estimate_whisper_quality


class WhisperQualityTests(unittest.TestCase):
    def test_negative_log_probability_is_not_sigmoided(self):
        result = estimate_whisper_quality(
            [{
                "avg_logprob": -0.4,
                "no_speech_prob": 0.05,
                "compression_ratio": 1.2,
            }],
            "A reasonable short transcript.",
        )
        expected = math.exp(-0.4) * 0.95
        self.assertAlmostEqual(result["quality_score"], round(expected, 3))
        self.assertGreater(result["quality_score"], 0.60)

    def test_low_log_probability_and_high_no_speech_reduce_score(self):
        result = estimate_whisper_quality(
            [{
                "avg_logprob": -1.8,
                "no_speech_prob": 0.8,
                "compression_ratio": 1.1,
            }],
            "Uncertain words.",
        )
        self.assertLess(result["quality_score"], 0.10)

    def test_empty_transcript_has_zero_quality(self):
        result = estimate_whisper_quality(
            [{"avg_logprob": -0.2, "no_speech_prob": 0.0}],
            "",
        )
        self.assertEqual(result["quality_score"], 0.0)
        self.assertIsNone(result["avg_logprob"])


if __name__ == "__main__":
    unittest.main()
