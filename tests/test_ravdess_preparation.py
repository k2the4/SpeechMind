import unittest
from pathlib import Path

from scripts.train_voice_emotion import (
    assign_speaker_splits,
    parse_ravdess_filename,
    summarize,
)


class RavdessPreparationTests(unittest.TestCase):
    def test_parses_ravdess_speech_and_maps_labels(self):
        row = parse_ravdess_filename(Path("03-01-05-01-02-01-12.wav"))
        self.assertEqual(row["label"], "anger")
        self.assertEqual(row["speaker_id"], "actor_12")

    def test_ignores_song_and_non_ravdess_files(self):
        self.assertIsNone(parse_ravdess_filename("03-02-05-01-02-01-12.wav"))
        self.assertIsNone(parse_ravdess_filename("sample.wav"))

    def test_actor_groups_do_not_leak_between_splits(self):
        rows = [
            {"path": f"{actor}-{emotion}", "speaker_id": f"actor_{actor:02d}",
             "label": "joy", "source_emotion_code": "03"}
            for actor in range(1, 25) for emotion in range(2)
        ]
        first = assign_speaker_splits(rows, seed=7)
        second = assign_speaker_splits(rows, seed=7)
        self.assertEqual([row["split"] for row in first],
                         [row["split"] for row in second])
        groups = {
            split: {row["speaker_id"] for row in first if row["split"] == split}
            for split in ("train", "validation", "test")
        }
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])

    def test_summary_reports_all_splits(self):
        rows = [
            {"speaker_id": "actor_01", "label": "joy", "split": "train"},
            {"speaker_id": "actor_02", "label": "anger", "split": "test"},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["train"]["files"], 1)
        self.assertEqual(summary["test"]["labels"], {"anger": 1})


if __name__ == "__main__":
    unittest.main()
