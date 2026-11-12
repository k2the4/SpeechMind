import unittest

from fusion_layer import fuse


class FusionReliabilityTests(unittest.TestCase):
    def test_unknown_asr_quality_uses_equal_branch_weights(self):
        result = fuse(
            {"emotion": "neutral", "distribution": {"neutral": 1.0}},
            {"emotion": "neutral", "distribution": {"neutral": 1.0}},
            asr_confidence=None,
        )

        self.assertEqual(result["fusion_weights"], {"text": 0.5, "voice": 0.5})
        self.assertEqual(result["interpretation_status"], "branch_agreement")
        self.assertIn("ASR quality indicator unavailable", result["uncertainty_reasons"])

    def test_ambiguous_voice_is_not_misreported_as_branch_disagreement(self):
        result = fuse(
            {"emotion": "neutral", "distribution": {"neutral": 1.0}},
            {
                "emotion": "fear",
                "distribution": {
                    "fear": 0.260,
                    "disgust": 0.257,
                    "sadness": 0.179,
                    "neutral": 0.128,
                    "joy": 0.096,
                    "anger": 0.063,
                    "surprise": 0.017,
                },
            },
            asr_confidence=None,
        )

        self.assertEqual(result["voice_emotion"], "uncertain")
        self.assertTrue(result["voice_label_ambiguous"])
        self.assertFalse(result["branch_disagreement"])
        self.assertIsNone(result["branch_agreement"])
        self.assertEqual(result["interpretation_status"], "single_branch")
        self.assertEqual(result["overall_emotion"], "neutral")
        self.assertEqual(result["fusion_weights"], {"text": 1.0, "voice": 0.0})
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["confidence_margin"])
        self.assertIn("inconclusive", result["branch_comparison"].lower())

    def test_ambiguous_voice_without_text_evidence_abstains(self):
        result = fuse(
            {"available": False, "emotion": "unavailable", "distribution": {}},
            {
                "emotion": "fear",
                "distribution": {"fear": 0.260, "disgust": 0.257, "sadness": 0.179, "neutral": 0.128, "joy": 0.096, "anger": 0.063, "surprise": 0.017},
            },
            asr_confidence=None,
        )

        self.assertEqual(result["interpretation_status"], "insufficient_evidence")
        self.assertEqual(result["overall_emotion"], "unresolved")
        self.assertTrue(result["voice_label_ambiguous"])
        self.assertIsNone(result["confidence"])

    def test_clear_text_voice_disagreement_still_abstains(self):
        result = fuse(
            {"emotion": "joy", "distribution": {"joy": 1.0}},
            {"emotion": "anger", "distribution": {"anger": 1.0}},
            asr_confidence=0.9,
        )

        self.assertTrue(result["branch_disagreement"])
        self.assertEqual(result["interpretation_status"], "branch_disagreement")
        self.assertEqual(result["overall_emotion"], "unresolved")
        self.assertIsNone(result["confidence"])


if __name__ == "__main__":
    unittest.main()
