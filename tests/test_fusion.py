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

    def test_unknown_voice_label_becomes_insufficient_evidence_not_neutral(self):
        result = fuse(
            text_result={"emotion": "unavailable", "available": False, "distribution": {}},
            voice_result={"emotion": "made_up_label", "distribution": {"made_up_label": 1.0}},
            asr_confidence=0.9,
        )
        self.assertEqual(result["interpretation_status"], "insufficient_evidence")
        self.assertEqual(result["overall_emotion"], "unavailable")
        self.assertNotEqual(result["overall_emotion"], "neutral")

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
        self.assertEqual(result["overall_emotion"], "unresolved")
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["valence"], "Unknown")
        self.assertTrue(result["branch_disagreement"])
        self.assertEqual(result["interpretation_status"], "branch_disagreement")
        self.assertIn("Text signals joy; vocal signals sadness", result["branch_comparison"])
        self.assertIn("No combined emotion has been assigned", result["overall_interpretation"])

    def test_near_tied_voice_scores_are_reported_as_uncertain(self):
        result = fuse(
            text_result={
                "emotion": "neutral",
                "distribution": {"neutral": 1.0},
            },
            voice_result={
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
        self.assertTrue(result["branch_disagreement"])
        self.assertEqual(result["overall_emotion"], "unresolved")
        self.assertIn("does not clearly separate fear", result["branch_comparison"])
        self.assertIn(
            "Voice model top-two scores are too close to support a stable label",
            result["uncertainty_reasons"],
        )

    def test_both_branches_unavailable_returns_insufficient_evidence(self):
        result = fuse(
            text_result={"emotion": "unavailable", "available": False},
            voice_result={"emotion": "unavailable", "available": False},
            asr_confidence=None,
        )
        self.assertEqual(result["interpretation_status"], "insufficient_evidence")
        self.assertEqual(result["overall_emotion"], "unavailable")
        self.assertEqual(result["fusion_weights"], {"text": 0.0, "voice": 0.0})
        self.assertEqual(result["fused_distribution"], {})


if __name__ == "__main__":
    unittest.main()
