import csv
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_speechmind_manifest import (
    classification_metrics,
    read_manifest,
    select_records,
    speaker_split_report,
    unseen_speaker_records,
)


class SpeechMindManifestEvaluationTests(unittest.TestCase):
    def _row(self, speaker, split="test", age_group="adult", dataset="hiacc"):
        return {
            "audio_path": f"{speaker}.wav",
            "transcript": "यह एक test है",
            "speaker_id": speaker,
            "split": split,
            "age_group": age_group,
            "dataset": dataset,
        }

    def test_manifest_reader_requires_speaker_metadata_and_selects_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["audio_path", "transcript", "speaker_id", "split"],
                )
                writer.writeheader()
                writer.writerow({
                    "audio_path": "a.wav",
                    "transcript": "hello",
                    "speaker_id": "speaker-a",
                    "split": "train",
                })
                writer.writerow({
                    "audio_path": "b.wav",
                    "transcript": "नमस्ते",
                    "speaker_id": "speaker-b",
                    "split": "test",
                })
            all_rows, test_rows = read_manifest(path, "test")
            self.assertEqual(len(all_rows), 2)
            self.assertEqual([row["audio_path"] for row in test_rows], ["b.wav"])

    def test_speaker_split_report_detects_overlap(self):
        rows = [
            self._row("speaker-a", "train"),
            self._row("speaker-b", "validation"),
            self._row("speaker-a", "test"),
            self._row("speaker-c", "test"),
        ]
        report = speaker_split_report(rows, "test")
        self.assertEqual(report["status"], "speaker_overlap")
        self.assertEqual(report["overlapping_speaker_count"], 1)
        self.assertEqual(report["unseen_evaluation_speaker_count"], 1)

    def test_unseen_speaker_filter_excludes_train_and_validation_speakers(self):
        rows = [
            self._row("speaker-a", "train"),
            self._row("speaker-b", "validation"),
            self._row("speaker-a", "test"),
            self._row("speaker-c", "test"),
        ]
        evaluation = [row for row in rows if row["split"] == "test"]
        selected = unseen_speaker_records(rows, evaluation)
        self.assertEqual([row["speaker_id"] for row in selected], ["speaker-c"])

    def test_sampler_is_reproducible_and_respects_speaker_cap(self):
        rows = [
            self._row("a1", age_group="adult"),
            self._row("a1", age_group="adult"),
            self._row("a2", age_group="adult"),
            self._row("c1", age_group="children"),
            self._row("c2", age_group="children"),
            self._row("c3", age_group="children"),
        ]
        first = select_records(rows, limit=4, seed=7, max_per_speaker=1)
        second = select_records(rows, limit=4, seed=7, max_per_speaker=1)
        self.assertEqual(
            [row["audio_path"] for row in first],
            [row["audio_path"] for row in second],
        )
        speakers = [row["speaker_id"] for row in first]
        self.assertEqual(len(speakers), len(set(speakers)))
        groups = [row["age_group"] for row in first]
        self.assertIn("adult", groups)
        self.assertIn("children", groups)

    def test_emotion_metrics_report_abstention_separately_from_accuracy(self):
        rows = [
            {"emotion_label": "joy", "voice_emotion": "joy"},
            {"emotion_label": "fear", "voice_emotion": "uncertain"},
            {"emotion_label": "anger", "voice_emotion": "sad"},
        ]
        result = classification_metrics(rows, "voice_emotion")
        self.assertEqual(result["labelled_samples"], 3)
        self.assertEqual(result["covered_samples"], 2)
        self.assertEqual(result["abstained_samples"], 1)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        self.assertAlmostEqual(result["accuracy_all_labelled_samples"], 1 / 3)
        self.assertAlmostEqual(result["accuracy_on_covered_samples"], 0.5)

    def test_full_runner_writes_asr_translation_emotion_and_timing_scores(self):
        import json
        import sys
        import types
        from unittest.mock import patch

        from scripts.evaluate_speechmind_manifest import run_evaluation

        class FakePipeline:
            def __init__(self):
                pass

            def run(self, _audio_path):
                return {
                    "transcript": "hello world",
                    "meaning_en": "Hello world.",
                    "meaning_en_available": True,
                    "text_emotion": "joy",
                    "voice_emotion": "joy",
                    "overall_emotion": "joy",
                    "interpretation_status": "branch_agreement",
                    "asr_model": "fake-local-asr",
                    "timings_sec": {"asr": 1.0, "translation": 2.0},
                    "processing_time_sec": 3.0,
                }

        fake_pipeline_module = types.ModuleType("pipeline")
        fake_pipeline_module.SpeechMindPipeline = FakePipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_root = root / "audio"
            audio_root.mkdir()
            (audio_root / "clip.wav").write_bytes(b"test")
            manifest = root / "manifest.csv"
            fields = [
                "audio_path", "transcript", "speaker_id", "split",
                "dataset", "age_group", "reference_meaning_en", "emotion_label",
            ]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "audio_path": "clip.wav",
                    "transcript": "hello world",
                    "speaker_id": "speaker-test",
                    "split": "test",
                    "dataset": "tiny-fixture",
                    "age_group": "adult",
                    "reference_meaning_en": "Hello world.",
                    "emotion_label": "joy",
                })
                writer.writerow({
                    "audio_path": "train.wav",
                    "transcript": "training only",
                    "speaker_id": "speaker-train",
                    "split": "train",
                    "dataset": "tiny-fixture",
                    "age_group": "adult",
                })
            output_dir = root / "output"
            with patch.dict(sys.modules, {"pipeline": fake_pipeline_module}):
                summary = run_evaluation(
                    manifest_path=manifest,
                    audio_root=audio_root,
                    output_dir=output_dir,
                    split="test",
                    limit=0,
                    max_per_speaker=1,
                )

            self.assertEqual(summary["asr"]["micro"]["wer"], 0.0)
            self.assertEqual(summary["translation"]["micro"]["wer"], 0.0)
            self.assertEqual(summary["emotion"]["voice"]["accuracy_all_labelled_samples"], 1.0)
            self.assertEqual(summary["emotion"]["text"]["accuracy_all_labelled_samples"], 1.0)
            self.assertEqual(summary["performance"]["median_stage_timings"]["asr"]["median_seconds"], 1.0)
            self.assertTrue((output_dir / "predictions.csv").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            saved = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["samples"]["successful_pipeline_runs"], 1)

    def test_no_gold_emotion_labels_means_not_scored(self):
        self.assertIsNone(classification_metrics(
            [{"emotion_label": "", "voice_emotion": "neutral"}],
            "voice_emotion",
        ))


if __name__ == "__main__":
    unittest.main()
