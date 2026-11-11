import unittest

from scripts.benchmark_srota_hiacc import DATA_ROOT, resolve_audio_path, select_records, script_label


class SrotaBenchmarkTests(unittest.TestCase):
    def test_manifest_audio_path_resolves_under_hiacc_data_root(self):
        row = {"audio_path": "raw/Corpus/adult/audio/test_split/AD63063.wav"}
        self.assertEqual(
            resolve_audio_path(row),
            DATA_ROOT / "raw/Corpus/adult/audio/test_split/AD63063.wav",
        )

    def test_sample_is_balanced_across_adult_and_children(self):
        rows = []
        for group in ("adult", "children"):
            for index in range(10):
                rows.append({
                    "age_group": group,
                    "speaker_id": f"{group}-{index}",
                    "audio_filename": f"{group}-{index}.wav",
                })
        selected = select_records(rows, limit=8, seed=42, max_per_speaker=1)
        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(row["age_group"] == "adult" for row in selected), 4)
        self.assertEqual(sum(row["age_group"] == "children" for row in selected), 4)

    def test_one_clip_per_speaker_limit(self):
        rows = [
            {"age_group": "adult", "speaker_id": "A", "audio_filename": "a1.wav"},
            {"age_group": "adult", "speaker_id": "A", "audio_filename": "a2.wav"},
            {"age_group": "adult", "speaker_id": "B", "audio_filename": "b1.wav"},
            {"age_group": "children", "speaker_id": "C", "audio_filename": "c1.wav"},
        ]
        selected = select_records(rows, limit=10, seed=1, max_per_speaker=1)
        self.assertEqual(len(selected), 3)
        speakers = [row["speaker_id"] for row in selected]
        self.assertEqual(len(speakers), len(set(speakers)))

    def test_native_hinglish_output_is_detected_as_mixed_script(self):
        self.assertEqual(
            script_label("मेरा favourite festival Diwali है"),
            "Devanagari+Latin",
        )


if __name__ == "__main__":
    unittest.main()
