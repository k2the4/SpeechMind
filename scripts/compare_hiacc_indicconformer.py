"""Compare SpeechMind's Hindi IndicConformer route on the exact Srota HiACC smoke-test clips.

This is a diagnostic cross-language stress test, not a claim that IndicConformer
is designed for Hinglish. It deliberately uses the exact clip list and references
saved by the Srota benchmark so the two CSVs can be compared row by row.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.asr import SpeechRecognizer
from scripts.asr_metrics import aggregate_error_counts, error_counts_multi_reference
from utils.audio_utils import load_audio

DATA_ROOT = PROJECT_ROOT / "data" / "hiacc"
DEFAULT_INPUT = DATA_ROOT / "benchmarks" / "srota" / "predictions.csv"
DEFAULT_OUTPUT = DATA_ROOT / "benchmarks" / "indicconformer"


def resolve_audio_path(audio_path: str) -> Path:
    return DATA_ROOT / audio_path


def read_srota_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No Srota predictions found in {path}")
    required = {"audio_path", "reference"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", default="hi")
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        records = read_srota_predictions(args.input)
        print(
            f"[IndicConformer] Exact Srota clip list: {len(records)} clips; "
            f"forced language={args.language}"
        )
        print(
            "[IndicConformer] Diagnostic only: this Hindi model is not a "
            "Hinglish-specialist candidate."
        )

        init_started = time.perf_counter()
        recognizer = SpeechRecognizer()
        print("[IndicConformer] Loading IndicConformer...")
        recognizer._load_indic_model()
        initialization_seconds = time.perf_counter() - init_started
        print(f"[IndicConformer] Model initialization: {initialization_seconds:.2f}s")

        rows = []
        metric_rows = []
        inference_times = []
        inference_started = time.perf_counter()

        for index, record in enumerate(records, start=1):
            audio_path = resolve_audio_path(record["audio_path"])
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing HiACC audio file: {audio_path}")
            audio = load_audio(str(audio_path))
            clip_started = time.perf_counter()
            hypothesis = recognizer._indic_transcribe(audio, args.language)
            inference_seconds = time.perf_counter() - clip_started

            reference = str(record.get("reference") or "")
            metrics = error_counts_multi_reference(reference, hypothesis)
            metric_rows.append(metrics)
            inference_times.append(inference_seconds)
            rows.append({
                "audio_path": record["audio_path"],
                "audio_filename": audio_path.name,
                "age_group": record.get("age_group"),
                "speaker_id": record.get("speaker_id"),
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "inference_seconds": round(inference_seconds, 3),
            })
            if index % 5 == 0 or index == len(records):
                current = aggregate_error_counts(metric_rows)
                print(
                    f"[IndicConformer] {index}/{len(records)} complete; "
                    f"last={inference_seconds:.1f}s; WER so far={current['wer']:.3f}"
                )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = args.output_dir / "predictions.csv"
        with predictions_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "dataset": "HiACC",
            "split": "test",
            "model": "ai4bharat/indic-conformer-600m-multilingual",
            "language_forced": args.language,
            "device": str(recognizer.device),
            "samples": len(rows),
            "overall": aggregate_error_counts(metric_rows),
            "model_initialization_seconds": round(initialization_seconds, 3),
            "clip_inference_elapsed_seconds": round(
                time.perf_counter() - inference_started, 3
            ),
            "mean_inference_seconds": round(statistics.mean(inference_times), 3),
            "median_inference_seconds": round(statistics.median(inference_times), 3),
            "total_elapsed_seconds": round(time.perf_counter() - started, 3),
            "input_predictions_csv": str(args.input.resolve()),
            "predictions_csv": str(predictions_path.resolve()),
            "notes": [
                "Uses the exact audio_path and reference rows from the Srota smoke-test CSV.",
                "Forced Hindi IndicConformer is a cross-language diagnostic, not a Hinglish-specialist comparison.",
                "No Whisper fallback is used; initialization or inference errors stop the run.",
                "Metrics use the same SpeechMind NFKC/lowercase/punctuation normalization utility as Srota.",
                "The 20-clip result is diagnostic only and is not production evidence.",
            ],
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\n[IndicConformer] Benchmark complete.")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"\n[IndicConformer] ERROR: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
