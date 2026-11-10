import csv
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_hiacc_dataset import (
    _domain,
    _parse_transcript_file,
    _speaker_from_name,
    _split,
    safe_extract,
)


class HiACCPreparationTests(unittest.TestCase):
    def test_domain_and_split_detection(self):
        self.assertEqual(_domain(("HiACC", "Adult", "audio", "test", "a.wav")), "adult")
        self.assertEqual(_domain(("HiACC", "Children", "transcripts", "val")), "children")
        self.assertEqual(_split(("Adult", "audio", "train", "a.wav")), "train")
        self.assertEqual(_split(("Children", "transcripts", "validation")), "val")

    def test_speaker_id_is_extracted_when_present(self):
        self.assertEqual(_speaker_from_name("speaker_ABC123_utt01.wav"), "ABC123")
        self.assertEqual(_speaker_from_name("ordinary_clip.wav"), "")

    def test_csv_transcripts_preserve_commas_and_mixed_scripts(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transcripts.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["audio_file", "transcript"])
                writer.writerow([
                    "Adult/audio/test/sample_001.wav",
                    "मैंने my favourite song सुना, actually बहुत अच्छा था",
                ])
            rows = _parse_transcript_file(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "sample_001.wav")
            self.assertEqual(
                rows[0][1],
                "मैंने my favourite song सुना, actually बहुत अच्छा था",
            )

    def test_mirrored_single_text_file_is_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample_001.txt"
            path.write_text("आज I went to college", encoding="utf-8")
            rows = _parse_transcript_file(path)
            self.assertEqual(rows, [("sample_001.wav", "आज I went to college")])

    def test_zip_extraction_rejects_path_traversal(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../escape.txt", "not allowed")
            with self.assertRaisesRegex(RuntimeError, "Unsafe path"):
                safe_extract(archive, root / "out")


if __name__ == "__main__":
    unittest.main()
