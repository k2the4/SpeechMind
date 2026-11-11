"""Recheck only HiACC clips previously classified as English by SpeechMind.

This narrowly validates the transcript-triggered Hinglish second pass without
repeating the completed 20-clip Srota benchmark or the full automatic run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["SPEECHMIND_ASR_PROVIDER"] = "local"

from models.asr import SpeechRecognizer
from scripts.asr_metrics import aggregate_error_counts, error_counts_multi_reference
from utils.audio_utils import load_audio

DATA_ROOT = PROJECT_ROOT / "data" / "hiacc"
DEFAULT_INPUT = (
    DATA_ROOT / "benchmarks" / "speechmind_auto" / "predictions.csv"
)
DEFAULT_OUTPUT = (
    DATA_ROOT / "benchmarks" / "english_hinglish_router"
)


def read_english_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No prior automatic-ASR records found in {path}")
    required = {"audio_path", "detected_language", "reference", "hypothesis"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    selected = [
        row for row in rows
        if str(row.get("detected_language") or "").strip().lower()
        in {"en", "english"}
    ]
    if not selected:
        raise ValueError(
            "The prior automatic-ASR CSV contains no English-detected clips. "
            "Run the original HiACC automatic benchmark once or inspect its CSV."
        )
    return selected


def _recompute_saved_metrics(output_dir: Path, input_path: Path) -> int:
    """Refresh scores in an existing predictions CSV without running ASR."""
    predictions_path = output_dir / "predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Saved predictions CSV not found: {predictions_path}. "
            "Run the normal route check once before using --metrics-only."
        )

    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No saved predictions found in {predictions_path}")

    previous_metrics = []
    updated_metrics = []
    inference_times = []
    routes = Counter()
    for row in rows:
        old = error_counts_multi_reference(
            row.get("reference", ""), row.get("previous_hypothesis", "")
        )
        new = error_counts_multi_reference(
            row.get("reference", ""), row.get("updated_hypothesis", "")
        )
        previous_metrics.append(old)
        updated_metrics.append(new)
        row.update({
            "reference_alternatives": new["reference_alternatives"],
            "previous_wer": old["wer"],
            "previous_cer": old["cer"],
            "updated_wer": new["wer"],
            "updated_cer": new["cer"],
            "previous_wer_reference_index": old["wer_reference_index"],
            "previous_cer_reference_index": old["cer_reference_index"],
            "updated_wer_reference_index": new["wer_reference_index"],
            "updated_cer_reference_index": new["cer_reference_index"],
        })
        routes[str(row.get("updated_model") or "unknown")] += 1
        try:
            inference_times.append(float(row.get("inference_seconds") or 0.0))
        except (TypeError, ValueError):
            pass

    fields = list(rows[0])
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "HiACC",
        "split": "test",
        "samples": len(rows),
        "selected_from": str(input_path.resolve()),
        "selection": "Saved predictions from previously English-detected clips",
        "evaluation_method": "best matching HiACC reference alternative; WER and CER selected independently",
        "previous_automatic_route": aggregate_error_counts(previous_metrics),
        "updated_automatic_route": aggregate_error_counts(updated_metrics),
        "updated_model_routes": dict(routes),
        "mean_inference_seconds": round(statistics.mean(inference_times), 3) if inference_times else None,
        "median_inference_seconds": round(statistics.median(inference_times), 3) if inference_times else None,
        "elapsed_seconds": 0.0,
        "predictions_csv": str(predictions_path.resolve()),
        "notes": [
            "Metrics were recalculated from saved hypotheses; no audio model inference was run.",
            "Reference alternatives separated by three spaces, tabs, or || are evaluated separately.",
            "WER and CER select the best matching alternative independently.",
            "This is a targeted regression check, not a representative benchmark.",
            "A small sample cannot establish production accuracy or recall.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[English/Hinglish route check] Recalculated saved hypotheses; ASR was not run.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Recalculate metrics from the existing predictions CSV without ASR inference.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        if args.metrics_only:
            return _recompute_saved_metrics(args.output_dir, args.input)

        records = read_english_records(args.input)
        print(
            f"[English/Hinglish route check] Rechecking {len(records)} "
            "previously English-detected clips only."
        )
        recognizer = SpeechRecognizer()

        rows = []
        previous_metrics = []
        updated_metrics = []
        inference_times = []
        routes: dict[str, int] = {}

        for index, record in enumerate(records, start=1):
            audio_path = DATA_ROOT / record["audio_path"]
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing HiACC audio file: {audio_path}")

            audio = load_audio(str(audio_path))
            clip_started = time.perf_counter()
            result = recognizer.transcribe(audio)
            inference_seconds = time.perf_counter() - clip_started

            old_metrics = error_counts_multi_reference(
                str(record.get("reference") or ""),
                str(record.get("hypothesis") or ""),
            )
            new_metrics = error_counts_multi_reference(
                str(record.get("reference") or ""),
                str(result.get("transcript") or ""),
            )
            previous_metrics.append(old_metrics)
            updated_metrics.append(new_metrics)
            inference_times.append(inference_seconds)

            model_used = str(result.get("asr_model") or "unknown")
            routes[model_used] = routes.get(model_used, 0) + 1
            rows.append({
                "audio_path": record["audio_path"],
                "reference": record["reference"],
                "reference_alternatives": new_metrics["reference_alternatives"],
                "previous_wer_reference_index": old_metrics["wer_reference_index"],
                "previous_cer_reference_index": old_metrics["cer_reference_index"],
                "updated_wer_reference_index": new_metrics["wer_reference_index"],
                "updated_cer_reference_index": new_metrics["cer_reference_index"],
                "previous_model": record.get("asr_model"),
                "previous_hypothesis": record.get("hypothesis"),
                "previous_wer": old_metrics["wer"],
                "previous_cer": old_metrics["cer"],
                "updated_model": model_used,
                "updated_language_source": result.get("language_source"),
                "updated_hypothesis": result.get("transcript"),
                "updated_languages_present": ",".join(
                    result.get("languages_present") or []
                ),
                "updated_romanized_hindi_detected": result.get(
                    "romanized_hindi_detected", False
                ),
                "updated_code_switch_detected": result.get(
                    "code_switch_detected", False
                ),
                "updated_wer": new_metrics["wer"],
                "updated_cer": new_metrics["cer"],
                "inference_seconds": round(inference_seconds, 3),
            })
            print(
                f"[English/Hinglish route check] {index}/{len(records)} | "
                f"{model_used} | {inference_seconds:.1f}s"
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
            "samples": len(rows),
            "selected_from": str(args.input.resolve()),
            "selection": "Only rows whose previous detected_language was en/english",
            "previous_automatic_route": aggregate_error_counts(previous_metrics),
            "updated_automatic_route": aggregate_error_counts(updated_metrics),
            "updated_model_routes": routes,
            "mean_inference_seconds": round(
                statistics.mean(inference_times), 3
            ),
            "median_inference_seconds": round(
                statistics.median(inference_times), 3
            ),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "predictions_csv": str(predictions_path.resolve()),
            "evaluation_method": "best matching HiACC reference alternative; WER and CER selected independently",
            "notes": [
                "Reference alternatives separated by three spaces, tabs, or || are evaluated separately.",
                "This is a targeted regression check, not a representative benchmark.",
                "The same saved English-detected audio paths and references are reused.",
                "Metrics are directly comparable only for these selected clips.",
                "A small sample cannot establish production accuracy or recall.",
            ],
        }
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\n[English/Hinglish route check] Complete.")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(
            f"\n[English/Hinglish route check] ERROR: "
            f"{type(error).__name__}: {error}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
