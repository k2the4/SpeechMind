import csv
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_hiacc_dataset import (
    _domain,
    _parse_transcript_file,
    _speaker_from_name,
    _speaker_metadata_map,
    _split,
    safe_extract,
)


class HiACCPreparationTests(unittest.TestCase):
    def test_domain_and_split_detection(self):
        self.assertEqual(_domain(("HiACC", "Adult", "audio", "test", "a.wav")), "adult")
        self.assertEqual(_domain(("HiACC", "Children", "transcripts", "val")), "children")
        self.assertEqual(_split(("Adult", "audio", "train", "a.wav")), "train")
        self.assertEqual(_split(("Children", "transcripts", "validation")), "val")
        self.assertEqual(
            _split(("Corpus", "adult", "audio", "test_split", "AD09001.wav")),
            "test",
        )
        self.assertEqual(
            _split(("Corpus", "adult", "audio", "train_split", "AD09002.wav")),
            "train",
        )
        self.assertEqual(
            _split(("Corpus", "children", "audio", "val_split", "CH03004.wav")),
            "val",
        )
        self.assertEqual(_split(("Adult", "Audio", "Training_Data", "a.wav")), "train")
        self.assertEqual(_split(("Adult", "Audio", "Testing", "a.wav")), "test")
        self.assertEqual(_split(("Adult", "Audio", "validation_set", "a.wav")), "val")

    def test_speaker_id_is_extracted_when_present(self):
        self.assertEqual(_speaker_from_name("speaker_ABC123_utt01.wav"), "ABC123")
        self.assertEqual(_speaker_from_name("AD63012.wav"), "AD63")
        self.assertEqual(_speaker_from_name("CH03004.wav"), "CH03")
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

    def test_hiacc_adult_transcription_line_format(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "combined_output_changed_test_output.txt"
            path.write_text(
                "AD30129.wav, जब वो गुफा तक पहुँचे\n"
                "AD21007.wav, better की hometown पर रहो और बाकी feel करो",
                encoding="utf-8",
            )
            rows = _parse_transcript_file(path)
            self.assertEqual(rows[0], ("AD30129.wav", "जब वो गुफा तक पहुँचे"))
            self.assertEqual(
                rows[1],
                ("AD21007.wav", "better की hometown पर रहो और बाकी feel करो"),
            )

    def test_zip_extraction_rejects_path_traversal(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../escape.txt", "not allowed")
            with self.assertRaisesRegex(RuntimeError, "Unsafe path"):
                safe_extract(archive, root / "out")



    def test_speaker_info_pid_maps_audio_filename_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_dir = root / "Corpus" / "adult" / "audio" / "test_split"
            audio_dir.mkdir(parents=True)
            (audio_dir / "AD09001.wav").write_bytes(b"")
            (audio_dir / "AD09002.wav").write_bytes(b"")
            info_dir = root / "Corpus" / "adult" / "metadata"
            info_dir.mkdir(parents=True)
            info = info_dir / "speaker_info.csv"
            info.write_text(
                "PID,Gender,Age,L1\nAD09,F,36,Hindi\n",
                encoding="utf-8",
            )
            mapping = _speaker_metadata_map(root)
            self.assertEqual(mapping["ad09001.wav"], "AD09")
            self.assertEqual(mapping["ad09002.wav"], "AD09")

    def test_speaker_metadata_is_joined_by_audio_filename(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = root / "Adult" / "metadata"
            metadata.mkdir(parents=True)
            stats = metadata / "sentence_stats.csv"
            with stats.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["audio_file", "speaker_id", "word_count"]
                )
                writer.writeheader()
                writer.writerow({
                    "audio_file": "Adult/audio/test/clip_001.wav",
                    "speaker_id": "ADULT_07",
                    "word_count": 8,
                })
            mapping = _speaker_metadata_map(root)
            self.assertEqual(mapping["clip_001.wav"], "ADULT_07")

if __name__ == "__main__":
    unittest.main()
