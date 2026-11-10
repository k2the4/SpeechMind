"""Benchmark local SpeechMind ASR against an IndicVoices Hindi sample."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import statistics
from collections import defaultdict
from pathlib import Path

# When this file is launched directly as
#   python scripts/evaluate_indicvoices_asr.py
# Python puts scripts/ (not the repository root) on sys.path. Add the
# repository root before importing the top-level models/ and utils/ packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent an inherited shell environment from accidentally using a metered API.
os.environ["SPEECHMIND_ASR_PROVIDER"] = "local"

from models.asr import SpeechRecognizer
from scripts.asr_metrics import aggregate_error_counts, error_counts
from utils.audio_utils import load_audio


DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "data/indicvoices/hindi/valid_sample"


def read_manifest(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def transcribe_one(
    recognizer: SpeechRecognizer, audio, model: str
) -> tuple[str, str, bool, str | None]:
    if model == "auto":
        result = recognizer.transcribe(audio)
        return (
            str(result.get("transcript") or ""),
            str(result.get("asr_model") or "auto"),
            bool(result.get("asr_fallback", False)),
            result.get("asr_fallback_reason"),
        )
    if model == "whisper":
        raw = recognizer._decode_with_whisper(audio, language=None)
        result = recognizer._whisper_result(raw, language_source="benchmark_whisper_auto")
        return str(result.get("transcript") or ""), "whisper-small", False, None
    if model == "indicconformer":
        result = recognizer._indic_result_with_whisper_fallback(audio, "hi")
        return (
            str(result.get("transcript") or ""),
            str(result.get("asr_model") or "indic-conformer-600m"),
            bool(result.get("asr_fallback", False)),
            result.get("asr_fallback_reason"),
        )
    raise ValueError(f"Unknown ASR model: {model}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument(
        "--model", choices=("auto", "whisper", "indicconformer"), default="auto"
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of records.")
    args = parser.parse_args()

    manifest_path = args.sample_dir / "manifest.jsonl"
    try:
        records = read_manifest(manifest_path)
        if args.limit > 0:
            records = records[:args.limit]

        print(f"[ASR Benchmark] Model={args.model}; clips={len(records)}")
        print("[ASR Benchmark] Local inference only; no API provider is used.")
        total_started = time.perf_counter()
        initialization_started = time.perf_counter()
        recognizer = SpeechRecognizer()

        # The IndicConformer model is lazy-loaded. For Hindi-split benchmarking,
        # warm it before timing individual clips so model loading/download time
        # does not get attributed to the first recording.
        indicconformer_warmup_error = None
        if args.model in ("auto", "indicconformer"):
            print("[ASR Benchmark] Initializing IndicConformer before clip timing...")
            try:
                recognizer._load_indic_model()
            except Exception as error:
                indicconformer_warmup_error = f"{type(error).__name__}: {error}"
                print(
                    "[ASR Benchmark] IndicConformer initialization failed; "
                    "per-clip fallback will use Whisper where applicable."
                )
        initialization_seconds = time.perf_counter() - initialization_started
        print(
            f"[ASR Benchmark] Model initialization: {initialization_seconds:.1f}s "
            "(reported separately)"
        )
        inference_started = time.perf_counter()

        rows = []
        normalized_metrics = []
        verbatim_metrics = []
        scenario_groups = defaultdict(list)

        for index, record in enumerate(records, start=1):
            audio_path = args.sample_dir / record["audio_file"]
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing audio file: {audio_path}")

            audio = load_audio(str(audio_path))
            started = time.perf_counter()
            hypothesis, actual_model, used_fallback, fallback_reason = transcribe_one(
                recognizer, audio, args.model
            )
            inference_seconds = time.perf_counter() - started

            normalized_reference = str(
                record.get("normalized") or record.get("text") or ""
            )
            verbatim_reference = str(
                record.get("verbatim") or normalized_reference
            )
            normalized_result = error_counts(normalized_reference, hypothesis)
            verbatim_result = error_counts(verbatim_reference, hypothesis)
            normalized_metrics.append(normalized_result)
            verbatim_metrics.append(verbatim_result)
            scenario = str(record.get("scenario") or "unknown")
            scenario_groups[scenario].append(normalized_result)

            rows.append({
                "sample_id": record.get("sample_id"),
                "speaker_id": record.get("speaker_id"),
                "scenario": scenario,
                "duration_seconds": record.get("duration"),
                "model_requested": args.model,
                "model_used": actual_model,
                "asr_fallback": used_fallback,
                "asr_fallback_reason": fallback_reason,
                "reference_normalized": normalized_reference,
                "reference_verbatim": verbatim_reference,
                "hypothesis": hypothesis,
                "wer_normalized": normalized_result["wer"],
                "cer_normalized": normalized_result["cer"],
                "wer_verbatim": verbatim_result["wer"],
                "cer_verbatim": verbatim_result["cer"],
                "inference_seconds": round(inference_seconds, 3),
            })

            if index % 5 == 0 or index == len(records):
                print(
                    f"[ASR Benchmark] {index}/{len(records)} complete; "
                    f"last={inference_seconds:.1f}s; "
                    f"WER so far={aggregate_error_counts(normalized_metrics)['wer']:.3f}"
                )

        output_dir = args.sample_dir / "benchmarks" / args.model
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "predictions.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        clip_elapsed = time.perf_counter() - inference_started
        elapsed = time.perf_counter() - total_started
        inference_values = [row["inference_seconds"] for row in rows]
        summary = {
            "dataset": "ai4bharat/IndicVoices",
            "config": "hindi",
            "split": "valid",
            "model_requested": args.model,
            "samples": len(rows),
            "fallback_count": sum(1 for row in rows if row["asr_fallback"]),
            "unique_speakers": len({row["speaker_id"] for row in rows}),
            "normalized_reference_metrics": aggregate_error_counts(normalized_metrics),
            "verbatim_reference_metrics": aggregate_error_counts(verbatim_metrics),
            "by_scenario": {
                scenario: aggregate_error_counts(group)
                for scenario, group in scenario_groups.items()
            },
            "model_initialization_seconds": round(initialization_seconds, 3),
            "indicconformer_warmup_error": indicconformer_warmup_error,
            "clip_inference_elapsed_seconds": round(clip_elapsed, 3),
            "mean_inference_seconds": round(
                statistics.mean(inference_values), 3
            ) if inference_values else 0.0,
            "median_inference_seconds": round(
                statistics.median(inference_values), 3
            ) if inference_values else 0.0,
            "total_elapsed_seconds": round(elapsed, 3),
            "predictions_csv": str(csv_path),
            "notes": [
                "WER and CER are computed after NFKC normalization, lowercasing, punctuation removal, and whitespace collapse.",
                "Normalized and verbatim references are reported separately.",
                "This is ASR evaluation only; IndicVoices has no emotion labels in this manifest.",
                "Model initialization is reported separately from per-clip inference time.",
                "Metrics apply only to this bounded Hindi validation sample, not to Hinglish or all Indian languages.",
            ],
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\n[ASR Benchmark] Complete.")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    except Exception as error:
        print(f"\n[ASR Benchmark] ERROR: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
