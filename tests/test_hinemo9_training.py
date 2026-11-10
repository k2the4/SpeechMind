import unittest

from scripts.train_hinemo9_text_emotion import (
    LABELS,
    LLM_LABEL_QUALITY,
    validate_rows,
    metrics_for,
)


class HinEmo9TrainingTests(unittest.TestCase):
    def test_model_derived_rows_are_down_weighted(self):
        rows = [
            {
                "text": "I am very happy",
                "label": "Joy",
                "label_source": "human_gold",
                "language_label": "English",
            },
            {
                "text": "Bahut khushi hui",
                "label": "Joy",
                "label_source": "llm",
                "language_label": "Hinglish",
            },
        ]
        texts, labels, weights, languages, skipped = validate_rows(rows, "train")
        self.assertEqual(len(texts), 2)
        self.assertEqual(labels, ["Joy", "Joy"])
        self.assertEqual(float(weights[0]), 1.0)
        self.assertAlmostEqual(float(weights[1]), 0.5 * LLM_LABEL_QUALITY["joy"])
        self.assertEqual(languages, ["english", "hinglish"])
        self.assertEqual(skipped, 0)

    def test_held_out_splits_reject_model_derived_labels(self):
        rows = [{
            "text": "I am happy",
            "label": "Joy",
            "label_source": "llm",
            "language_label": "English",
        }]
        with self.assertRaises(ValueError):
            validate_rows(rows, "test", require_gold=True)

    def test_unknown_labels_are_not_silently_dropped(self):
        rows = [{
            "text": "I feel peaceful",
            "label": "peace",
            "label_source": "human_gold",
            "language_label": "English",
        }]
        with self.assertRaises(ValueError):
            validate_rows(rows, "train")

    def test_metrics_keep_all_nine_emotions_in_macro_f1(self):
        truth = ["Joy", "Anger", "Neutral"]
        predicted = ["Joy", "Neutral", "Neutral"]
        result = metrics_for(truth, predicted)
        self.assertEqual(result["confusion_matrix"][2][8], 1)
        self.assertAlmostEqual(result["macro_f1"], 5.0 / 27.0, places=6)


if __name__ == "__main__":
    unittest.main()
