"""Benchmark SpeechMind's current automatic ASR route on the saved HiACC smoke-test clips.

Uses the exact audio paths and references from the Srota predictions CSV.
This measures the current Whisper language-detection -> IndicConformer/Whisper
routing behavior; it does not modify production routing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["SPEECHMIND_ASR_PROVIDER"] = "local"

from models.asr import SpeechRecognizer
from scripts.asr_metrics import aggregate_error_counts, error_counts_multi_reference
from utils.audio_utils import load_audio

DATA_ROOT = PROJECT_ROOT / "data" / "hiacc"
DEFAULT_INPUT = DATA_ROOT / "benchmarks" / "srota" / "predictions.csv"
DEFAULT_OUTPUT = DATA_ROOT / "benchmarks" / "speechmind_auto"


def resolve_audio_path(audio_path: str) -> Path:
    return DATA_ROOT / audio_path


def read_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No Srota predictions found in {path}")
    required = {"audio_path", "reference"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    return rows


def script_label(value: str) -> str:
    scripts = set()
    for char in str(value or ""):
        name = unicodedata.name(char, "")
        if "DEVANAGARI" in name:
            scripts.add("Devanagari")
        elif "LATIN" in name:
            scripts.add("Latin")
        elif "ARABIC" in name:
            scripts.add("Arabic")
    return "+".join(sorted(scripts)) if scripts else "empty_or_other"




def read_previous_predictions(path: Path) -> dict[str, dict]:
    """Load the previous automatic route before its output CSV is replaced."""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    previous = {}
    for row in rows:
        audio_path = str(row.get("audio_path") or "").replace("\\", "/").lower()
        if audio_path and row.get("hypothesis") is not None:
            previous[audio_path] = row
    return previous

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        records = read_records(args.input)
        predictions_path = args.output_dir / "predictions.csv"
        previous_predictions = read_previous_predictions(predictions_path)
        if previous_predictions:
            print(
                f"[SpeechMind Auto] Found {len(previous_predictions)} previous "
                "predictions; this run will compare old and updated routing."
            )
        else:
            print("[SpeechMind Auto] No previous output found; running a baseline.")
        print(f"[SpeechMind Auto] Exact Srota clip list: {len(records)} clips")
        print("[SpeechMind Auto] Running current automatic language detection and routing.")
        init_started = time.perf_counter()
        recognizer = SpeechRecognizer()
        whisper_initialization_seconds = time.perf_counter() - init_started
        print(
            f"[SpeechMind Auto] Whisper initialization: "
            f"{whisper_initialization_seconds:.2f}s"
        )

        rows = []
        metric_rows = []
        inference_times = []
        route_counts = {}
        script_counts = {}
        language_counts = {}
        by_age_group = {}
        previous_metric_rows = []
        comparison_rows = 0
        inference_started = time.perf_counter()

        for index, record in enumerate(records, start=1):
            audio_path = resolve_audio_path(record["audio_path"])
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing HiACC audio file: {audio_path}")
            audio = load_audio(str(audio_path))
            clip_started = time.perf_counter()
            result = recognizer.transcribe(audio)
            inference_seconds = time.perf_counter() - clip_started

            hypothesis = str(result.get("transcript") or "")
            reference = str(record.get("reference") or "")
            metrics = error_counts_multi_reference(reference, hypothesis)
            metric_rows.append(metrics)
            inference_times.append(inference_seconds)

            normalized_audio_path = str(record["audio_path"]).replace("\\", "/").lower()
            previous = previous_predictions.get(normalized_audio_path)
            previous_metrics = None
            if previous is not None and str(previous.get("reference") or "") == reference:
                previous_metrics = error_counts_multi_reference(
                    reference, str(previous.get("hypothesis") or "")
                )
                previous_metric_rows.append(previous_metrics)
                comparison_rows += 1
            elif previous is not None:
                print(
                    f"[SpeechMind Auto] Previous reference differs for "
                    f"{audio_path.name}; excluding that row from before/after comparison."
                )

            model_used = str(result.get("asr_model") or "unknown")
            route_counts[model_used] = route_counts.get(model_used, 0) + 1
            output_script = script_label(hypothesis)
            script_counts[output_script] = script_counts.get(output_script, 0) + 1
            detected_language = str(result.get("language") or "unknown")
            language_counts[detected_language] = language_counts.get(detected_language, 0) + 1
            age_group = str(record.get("age_group") or "unknown")
            by_age_group.setdefault(age_group, []).append(metrics)

            rows.append({
                "audio_path": record["audio_path"],
                "audio_filename": audio_path.name,
                "age_group": record.get("age_group"),
                "speaker_id": record.get("speaker_id"),
                "reference": reference,
                "previous_hypothesis": (
                    str(previous.get("hypothesis") or "") if previous_metrics else ""
                ),
                "previous_model": (
                    str(previous.get("asr_model") or "") if previous_metrics else ""
                ),
                "previous_wer": previous_metrics["wer"] if previous_metrics else None,
                "previous_cer": previous_metrics["cer"] if previous_metrics else None,
                "previous_wer_reference_index": (
                    previous_metrics["wer_reference_index"] if previous_metrics else None
                ),
                "previous_cer_reference_index": (
                    previous_metrics["cer_reference_index"] if previous_metrics else None
                ),
                "hypothesis": hypothesis,
                "detected_language": detected_language,
                "language_source": result.get("language_source"),
                "languages_present": ",".join(result.get("languages_present") or []),
                "code_switch_detected": result.get("code_switch_detected", False),
                "asr_model": model_used,
                "asr_fallback": result.get("asr_fallback", False),
                "asr_fallback_reason": result.get("asr_fallback_reason"),
                "hypothesis_script": output_script,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "wer_reference_index": metrics["wer_reference_index"],
                "cer_reference_index": metrics["cer_reference_index"],
                "wer_change_vs_previous": (
                    metrics["wer"] - previous_metrics["wer"] if previous_metrics else None
                ),
                "cer_change_vs_previous": (
                    metrics["cer"] - previous_metrics["cer"] if previous_metrics else None
                ),
                "inference_seconds": round(inference_seconds, 3),
            })

            if index % 5 == 0 or index == len(records):
                current = aggregate_error_counts(metric_rows)
                print(
                    f"[SpeechMind Auto] {index}/{len(records)} complete; "
                    f"last={inference_seconds:.1f}s; WER so far={current['wer']:.3f}"
                )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "dataset": "HiACC",
            "split": "test",
            "route": "SpeechMind current automatic ASR",
            "samples": len(rows),
            "overall": aggregate_error_counts(metric_rows),
            "previous_route_comparison_samples": comparison_rows,
            "previous_route_overall": (
                aggregate_error_counts(previous_metric_rows)
                if previous_metric_rows else None
            ),
            "route_metric_change": (
                {
                    "wer_percentage_points": round(
                        (aggregate_error_counts(metric_rows)["wer"]
                         - aggregate_error_counts(previous_metric_rows)["wer"]) * 100,
                        3,
                    ),
                    "cer_percentage_points": round(
                        (aggregate_error_counts(metric_rows)["cer"]
                         - aggregate_error_counts(previous_metric_rows)["cer"]) * 100,
                        3,
                    ),
                    "interpretation": "negative values indicate improvement",
                }
                if previous_metric_rows else None
            ),
            "by_age_group": {
                group: aggregate_error_counts(group_metrics)
                for group, group_metrics in by_age_group.items()
            },
            "detected_language_counts": language_counts,
            "model_routes": route_counts,
            "hypothesis_script_counts": script_counts,
            "whisper_initialization_seconds": round(whisper_initialization_seconds, 3),
            "clip_inference_elapsed_seconds": round(
                time.perf_counter() - inference_started, 3
            ),
            "mean_inference_seconds": round(statistics.mean(inference_times), 3),
            "median_inference_seconds": round(statistics.median(inference_times), 3),
            "total_elapsed_seconds": round(time.perf_counter() - started, 3),
            "input_predictions_csv": str(args.input.resolve()),
            "predictions_csv": str(predictions_path.resolve()),
            "notes": [
                "Uses the exact audio_path and reference rows from the completed Srota smoke test.",
                "If a prior speechmind_auto/predictions.csv exists, old and new hypotheses are scored against the same alternative-aware references.",
                "The previous output is read before predictions.csv is overwritten.",
                "WER/CER use the same alternative-aware metric utility as the other HiACC benchmarks.",
                "Per-age-group, detected-language, selected-model, script, and code-switch diagnostics are retained.",
                "This 20-clip result is diagnostic only and is not production evidence.",
            ],
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\n[SpeechMind Auto] Benchmark complete.")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"\n[SpeechMind Auto] ERROR: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
