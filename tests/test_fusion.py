import unittest

from fusion_layer import fuse


class FusionTests(unittest.TestCase):
    def test_unavailable_text_branch_uses_voice_only(self):
        result = fuse(
            text_result={
                "emotion": "unavailable",
                "available": False,
                "distribution": {},
            },
            voice_result={
                "emotion": "happy",
                "confidence": 0.9,
                "distribution": {"happy": 0.9, "sad": 0.1},
                "arousal": "High",
            },
            asr_confidence=0.2,
        )
        self.assertEqual(result["fusion_weights"], {"text": 0.0, "voice": 1.0})
        self.assertEqual(result["overall_emotion"], "joy")
        self.assertAlmostEqual(sum(result["fused_distribution"].values()), 1.0)
        self.assertEqual(result["interpretation_status"], "single_branch")
        self.assertFalse(result["arousal_available"])
        self.assertEqual(result["arousal"], "Not assessed")

    def test_low_asr_score_reduces_text_weight(self):
        result = fuse(
            text_result={"emotion": "anger", "distribution": {"anger": 1.0}},
            voice_result={"emotion": "happy", "distribution": {"happy": 1.0}},
            asr_confidence=0.2,
        )
        self.assertLess(result["fusion_weights"]["text"], 0.60)
        self.assertGreater(result["fusion_weights"]["voice"], 0.40)
        self.assertTrue(result["branch_disagreement"])
        self.assertEqual(result["interpretation_status"], "branch_disagreement")

    def test_missing_asr_quality_uses_conservative_text_weight(self):
        result = fuse(
            text_result={"emotion": "anger", "distribution": {"anger": 1.0}},
            voice_result={"emotion": "joy", "distribution": {"joy": 1.0}},
            asr_confidence=None,
        )
        self.assertEqual(result["fusion_weights"], {"text": 0.3, "voice": 0.7})
        self.assertTrue(result["low_confidence_warning"])
        self.assertIn("ASR quality indicator unavailable", result["uncertainty_reasons"])

    def test_unknown_voice_label_is_not_silently_neutral(self):
        with self.assertRaises(ValueError):
            fuse(
                text_result={"emotion": "unavailable", "available": False, "distribution": {}},
                voice_result={"emotion": "made_up_label", "distribution": {"made_up_label": 1.0}},
                asr_confidence=0.9,
            )

    def test_confidence_is_explicitly_uncalibrated(self):
        result = fuse(
            text_result={"emotion": "joy", "distribution": {"joy": 1.0}},
            voice_result={"emotion": "happy", "distribution": {"happy": 1.0}},
            asr_confidence=0.9,
        )
        self.assertFalse(result["confidence_is_calibrated"])
        self.assertTrue(result["branch_agreement"])

    def test_same_emotion_synonyms_are_treated_as_agreement(self):
        result = fuse(
            text_result={"emotion": "joy", "distribution": {"joy": 1.0}},
            voice_result={"emotion": "happy", "distribution": {"happy": 1.0}},
            asr_confidence=0.9,
        )
        self.assertEqual(result["branch_agreement"], True)
        self.assertEqual(result["interpretation_status"], "branch_agreement")

    def test_disagreement_is_not_hidden_by_fused_winner(self):
        result = fuse(
            text_result={"emotion": "joy", "distribution": {"joy": 1.0}},
            voice_result={"emotion": "sad", "distribution": {"sad": 1.0}},
            asr_confidence=0.95,
        )
        self.assertEqual(result["overall_emotion"], "joy")
        self.assertTrue(result["branch_disagreement"])
        self.assertIn("Text signals joy; vocal signals sadness", result["branch_comparison"])

    def test_both_branches_unavailable_raises(self):
        with self.assertRaises(ValueError):
            fuse(
                text_result={"emotion": "unavailable", "available": False},
                voice_result={"emotion": "unavailable", "available": False},
                asr_confidence=None,
            )


if __name__ == "__main__":
    unittest.main()
