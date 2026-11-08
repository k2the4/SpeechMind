import tempfile
import unittest
from pathlib import Path

from utils.dataset_audit import audit_local_datasets, parse_ravdess_filename


class DatasetAuditTests(unittest.TestCase):
    def test_parse_ravdess_speech_filename(self):
        parsed = parse_ravdess_filename("03-01-05-02-01-02-12.wav")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["emotion"], "angry")
        self.assertEqual(parsed["actor"], "12")
        self.assertIsNone(parse_ravdess_filename("03-02-05-02-01-02-12.wav"))
        self.assertIsNone(parse_ravdess_filename("not-ravdess.wav"))

    def test_inventory_counts_only_valid_ravdess_speech(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "03-01-01-01-01-01-01.wav").touch()
            (root / "03-01-03-02-02-01-02.wav").touch()
            (root / "03-02-03-02-02-01-02.wav").touch()
            (root / "Hindi").mkdir()
            (root / "Hindi" / "sample.wav").touch()

            report = audit_local_datasets([root])
            self.assertEqual(report["ravdess_speech_wavs"], 2)
            self.assertEqual(report["ravdess_emotion_counts"]["neutral"], 1)
            self.assertEqual(report["ravdess_emotion_counts"]["happy"], 1)
            self.assertEqual(report["hindi_folder_wavs"], 1)


if __name__ == "__main__":
    unittest.main()
