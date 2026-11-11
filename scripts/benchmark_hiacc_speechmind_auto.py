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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        records = read_records(args.input)
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

            model_used = str(result.get("asr_model") or "unknown")
            route_counts[model_used] = route_counts.get(model_used, 0) + 1
            output_script = script_label(hypothesis)
            script_counts[output_script] = script_counts.get(output_script, 0) + 1

            rows.append({
                "audio_path": record["audio_path"],
                "audio_filename": audio_path.name,
                "age_group": record.get("age_group"),
                "speaker_id": record.get("speaker_id"),
                "reference": reference,
                "hypothesis": hypothesis,
                "detected_language": result.get("language"),
                "asr_model": model_used,
                "asr_fallback": result.get("asr_fallback", False),
                "asr_fallback_reason": result.get("asr_fallback_reason"),
                "hypothesis_script": output_script,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "inference_seconds": round(inference_seconds, 3),
            })

            if index % 5 == 0 or index == len(records):
                current = aggregate_error_counts(metric_rows)
                print(
                    f"[SpeechMind Auto] {index}/{len(records)} complete; "
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
            "route": "SpeechMind current automatic ASR",
            "samples": len(rows),
            "overall": aggregate_error_counts(metric_rows),
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
                "Measures the existing automatic Whisper language detection and IndicConformer/Whisper routing without changing production code.",
                "The first clip routed to IndicConformer may include its lazy model initialization in that clip's inference time.",
                "WER/CER use the same SpeechMind normalization utility as the Srota and IndicConformer diagnostics.",
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
