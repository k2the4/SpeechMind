import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.train_bhaav_text_emotion import LABELS, load_bhaav_dataframe, _split_dataframe


class BhaavLoaderTests(unittest.TestCase):
    def test_group_split_keeps_stories_disjoint(self):
        rows = []
        for group_index in range(10):
            for label in LABELS:
                rows.append({
                    "text": f"नमूना वाक्य {group_index} {label}",
                    "label": label,
                    "group": f"story-{group_index}",
                })
        frame = pd.DataFrame(rows)
        train, validation, test, group_split = _split_dataframe(frame)
        self.assertTrue(group_split)
        train_groups = set(train["group"])
        validation_groups = set(validation["group"])
        test_groups = set(test["group"])
        self.assertFalse(train_groups & validation_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(validation_groups & test_groups)

    def test_loads_and_normalises_bhaav_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bhaav.csv"
            rows = [
                ("यह वाक्य गुस्से को दिखाता है।", "anger", "story-1"),
                ("यह वाक्य खुशी को दिखाता है।", "joy", "story-2"),
                ("यह वाक्य रहस्य दिखाता है।", "suspense", "story-3"),
                ("यह वाक्य दुख दिखाता है।", "sad", "story-4"),
                ("यह सामान्य वाक्य है।", "neutral", "story-5"),
                ("एक और गुस्से वाला वाक्य।", "anger", "story-6"),
                ("एक और खुशी वाला वाक्य।", "joy", "story-7"),
                ("एक और रहस्यमय वाक्य।", "suspense", "story-8"),
                ("एक और दुखी वाक्य।", "sadness", "story-9"),
                ("एक और सामान्य वाक्य।", "neutral", "story-10"),
                ("यह वाक्य गुस्से को दिखाता है।", "anger", "story-11"),
            ]
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["sentence", "emotion", "story_id"])
                writer.writerows(rows)

            frame, source_name = load_bhaav_dataframe(path)
            self.assertEqual(len(frame), 10)
            self.assertEqual(set(frame["label"]), set(LABELS))
            self.assertEqual(source_name, str(path))
            self.assertIn("sadness", set(frame["label"]))


if __name__ == "__main__":
    unittest.main()
