import unittest

from models.text_emotion_format import format_multilabel_scores


class TextEmotionFormatTests(unittest.TestCase):
    def test_no_class_crossing_its_threshold_produces_neutral(self):
        result = format_multilabel_scores(
            {"joy": 0.659, "surprise": 0.525, "sadness": 0.37},
            {"joy": 0.70, "surprise": 0.60, "sadness": 0.55},
        )
        self.assertEqual(result["emotion"], "neutral")
        self.assertEqual(result["distribution"], {"neutral": 1.0})
        self.assertAlmostEqual(result["multi_label_scores"]["joy"], 0.659)

    def test_class_crossing_threshold_becomes_primary(self):
        result = format_multilabel_scores(
            {"joy": 0.82, "sadness": 0.66},
            {"joy": 0.70, "sadness": 0.60},
        )
        self.assertEqual(result["emotion"], "joy")
        self.assertIn("sadness", result["secondary_emotions"])


if __name__ == "__main__":
    unittest.main()
