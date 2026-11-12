import csv
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_asr_models_paired import (
    aggregate_predictions,
    grouped_metrics,
    paired_model_comparison,
    paired_vs_production,
    read_records,
    select_records,
)


class PairedASRBenchmarkTests(unittest.TestCase):
    def _row(self, speaker, age, audio=None):
        return {
            "audio_path": audio or f"{speaker}.wav",
            "reference_transcript": "मुझे इस project के लिए",
            "speaker_id": speaker,
            "age_group": age,
            "dataset": "hiacc",
            "production_hypothesis": "मुझे इस project के लिए",
        }

    def test_reader_accepts_full_pipeline_prediction_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            fields = [
                "audio_path", "reference_transcript", "speaker_id",
                "age_group", "hypothesis_transcript", "asr_model",
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "audio_path": "raw/adult/audio/a.wav",
                    "reference_transcript": "hello there",
                    "speaker_id": "adult-1",
                    "age_group": "adult",
                    "hypothesis_transcript": "hello there",
                    "asr_model": "whisper-small",
                })
            rows = read_records(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["reference_transcript"], "hello there")
            self.assertEqual(rows[0]["production_hypothesis"], "hello there")
            self.assertEqual(rows[0]["dataset"], "hiacc")

    def test_reader_accepts_legacy_reference_column_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            fields = ["audio_path", "reference", "speaker_id"]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "audio_path": "a.wav", "reference": "hello", "speaker_id": "a"
                })
                writer.writerow({
                    "audio_path": "a.wav", "reference": "hello", "speaker_id": "a"
                })
            with self.assertRaisesRegex(ValueError, "Duplicate audio_path"):
                read_records(path)

    def test_selection_is_reproducible_and_speaker_disjoint(self):
        rows = [
            self._row("a1", "adult"),
            self._row("a1", "adult", audio="a1-second.wav"),
            self._row("a2", "adult"),
            self._row("c1", "children"),
            self._row("c2", "children"),
            self._row("c3", "children"),
        ]
        first = select_records(rows, limit=4, seed=9, max_per_speaker=1)
        second = select_records(rows, limit=4, seed=9, max_per_speaker=1)
        self.assertEqual(
            [row["audio_path"] for row in first],
            [row["audio_path"] for row in second],
        )
        speakers = [row["speaker_id"] for row in first]
        self.assertEqual(len(speakers), len(set(speakers)))
        self.assertIn("adult", [row["age_group"] for row in first])
        self.assertIn("children", [row["age_group"] for row in first])

    def test_aggregates_use_actual_status_columns_and_reference_totals(self):
        rows = [
            {
                "reference_transcript": "one two",
                "srota_auto_hypothesis": "one two",
                "srota_auto_status": "ok",
                "srota_auto_seconds": 2.0,
            },
            {
                "reference_transcript": "one two",
                "srota_auto_hypothesis": "one",
                "srota_auto_status": "ok",
                "srota_auto_seconds": 4.0,
            },
            {
                "reference_transcript": "one two",
                "srota_auto_hypothesis": "",
                "srota_auto_status": "error",
                "srota_auto_seconds": "",
            },
        ]
        result = aggregate_predictions(rows, "srota_auto_hypothesis")
        self.assertEqual(result["selected_samples"], 3)
        self.assertEqual(result["successful_samples"], 2)
        self.assertEqual(result["failed_samples"], 1)
        self.assertAlmostEqual(result["micro"]["wer"], 0.5)
        self.assertAlmostEqual(
            result["micro_successful_samples_only"]["wer"], 0.25
        )
        self.assertEqual(result["median_inference_seconds"], 3.0)
        self.assertEqual(
            grouped_metrics(rows, "srota_auto_hypothesis", "age_group"),
            {"unknown": result["micro"]},
        )

    def test_reader_refuses_to_invent_speaker_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-speaker.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["audio_path", "reference_transcript"]
                )
                writer.writeheader()
                writer.writerow({
                    "audio_path": "clip.wav",
                    "reference_transcript": "hello there",
                })
            with self.assertRaisesRegex(ValueError, "no speaker_id"):
                read_records(path)

    def test_context_comparison_scores_failures_as_empty_hypotheses(self):
        rows = [
            {
                "reference_transcript": "one two",
                "srota_auto_hypothesis": "one two",
                "srota_native_hypothesis": "one",
            },
            {
                "reference_transcript": "one two",
                "srota_auto_hypothesis": "",
                "srota_native_hypothesis": "one two",
            },
        ]
        result = paired_model_comparison(
            rows, "srota_auto_hypothesis", "srota_native_hypothesis"
        )
        self.assertEqual(result["paired_samples"], 2)
        self.assertEqual(result["candidate_better"], 1)
        self.assertEqual(result["candidate_worse"], 1)
        self.assertEqual(result["tied"], 0)
        self.assertEqual(result["mean_clip_wer_delta"], 0.25)

    def test_paired_comparison_uses_identical_clips_and_reports_win_tie_loss(self):
        rows = [
            {
                "reference_transcript": "one two",
                "production_hypothesis": "one",
                "srota_auto_hypothesis": "one two",
                "srota_auto_status": "ok",
            },
            {
                "reference_transcript": "one two",
                "production_hypothesis": "one two",
                "srota_auto_hypothesis": "one",
                "srota_auto_status": "ok",
            },
            {
                "reference_transcript": "one two",
                "production_hypothesis": "one",
                "srota_auto_hypothesis": "one",
                "srota_auto_status": "ok",
            },
        ]
        result = paired_vs_production(rows, "srota_auto_hypothesis")
        self.assertEqual(result["paired_samples"], 3)
        self.assertEqual(result["candidate_better"], 1)
        self.assertEqual(result["candidate_worse"], 1)
        self.assertEqual(result["tied"], 1)
        self.assertEqual(result["mean_clip_wer_delta_vs_production"], 0.0)


if __name__ == "__main__":
    unittest.main()
