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

    def test_low_asr_score_reduces_text_weight(self):
        result = fuse(
            text_result={"emotion": "anger", "distribution": {"anger": 1.0}},
            voice_result={"emotion": "happy", "distribution": {"happy": 1.0}},
            asr_confidence=0.2,
        )
        self.assertLess(result["fusion_weights"]["text"], 0.60)
        self.assertGreater(result["fusion_weights"]["voice"], 0.40)

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


if __name__ == "__main__":
    unittest.main()
