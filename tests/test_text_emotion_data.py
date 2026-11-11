import unittest

from models.text_emotion_format import format_multilabel_scores
from scripts.prepare_text_emotion_data import brighter_record, hinemo_record


class TextEmotionDataPreparationTests(unittest.TestCase):
    def test_brighter_keeps_six_labels_and_full_mask(self):
        row = {
            "text": "मुझे यह बहुत पसंद आया",
            "anger": 0, "disgust": 0, "fear": 0,
            "joy": 1, "sadness": 0, "surprise": 0,
        }
        result = brighter_record(row, "train")
        self.assertEqual(result["labels"], [0, 0, 0, 1, 0, 0])
        self.assertEqual(result["label_mask"], [1, 1, 1, 1, 1, 1])
        self.assertEqual(result["source_weight"], 1.0)

    def test_brighter_no_emotion_is_neutral_candidate(self):
        row = {"text": "आज मौसम ठीक है"}
        for label in ("anger", "disgust", "fear", "joy", "sadness", "surprise"):
            row[label] = 0
        result = brighter_record(row, "test")
        self.assertEqual(sum(result["labels"]), 0)

    def test_hinemo_masks_unlabelled_categories(self):
        result = hinemo_record(
            {"text": "bahut happy hoon", "gpt_emotion": "joy"}, "train"
        )
        self.assertEqual(result["labels"], [0, 0, 0, 1, 0, 0])
        self.assertEqual(result["label_mask"], [1, 1, 0, 1, 1, 0])
        self.assertEqual(result["source_weight"], 0.35)
        self.assertEqual(result["label_provenance"], "gpt_generated_dataset_label")

    def test_unknown_hinemo_label_is_rejected(self):
        with self.assertRaises(ValueError):
            hinemo_record({"text": "test", "gpt_emotion": "surprise"}, "train")


class MultilabelOutputTests(unittest.TestCase):
    def test_no_signal_selects_neutral(self):
        result = format_multilabel_scores({
            "anger": 0.1, "disgust": 0.05, "fear": 0.2,
            "joy": 0.4, "sadness": 0.1, "surprise": 0.1,
        })
        self.assertEqual(result["emotion"], "neutral")
        self.assertEqual(result["distribution"], {"neutral": 1.0})

    def test_multiple_active_labels_keep_secondary_signal(self):
        result = format_multilabel_scores({
            "anger": 0.82, "disgust": 0.05, "fear": 0.1,
            "joy": 0.76, "sadness": 0.02, "surprise": 0.1,
        })
        self.assertEqual(result["emotion"], "anger")
        self.assertEqual(result["secondary_emotions"], ["joy"])
        self.assertAlmostEqual(sum(result["distribution"].values()), 1.0)


if __name__ == "__main__":
    unittest.main()
