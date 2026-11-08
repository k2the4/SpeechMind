import unittest

from fusion_layer import (
    UNIFIED_EMOTIONS,
    TEXT_TO_UNIFIED,
    VOICE_TO_UNIFIED,
    fuse,
)


class EmotionMappingTests(unittest.TestCase):
    def test_mapping_targets_exist_in_unified_space(self):
        self.assertTrue(set(TEXT_TO_UNIFIED.values()).issubset(set(UNIFIED_EMOTIONS)))
        self.assertTrue(set(VOICE_TO_UNIFIED.values()).issubset(set(UNIFIED_EMOTIONS)))

    def test_bhaav_suspense_is_not_mapped_to_fear(self):
        self.assertEqual(TEXT_TO_UNIFIED["suspense"], "suspense")
        self.assertNotEqual(TEXT_TO_UNIFIED["suspense"], "fear")

    def test_suspense_survives_text_only_fusion(self):
        result = fuse(
            text_result={
                "emotion": "suspense",
                "available": True,
                "distribution": {"suspense": 1.0},
            },
            voice_result={
                "emotion": "unavailable",
                "available": False,
                "distribution": {},
            },
            asr_confidence=0.95,
        )
        self.assertEqual(result["overall_emotion"], "suspense")
        self.assertEqual(result["valence"], "Neutral")


if __name__ == "__main__":
    unittest.main()
