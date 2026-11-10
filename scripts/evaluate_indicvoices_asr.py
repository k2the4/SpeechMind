"""Benchmark local SpeechMind ASR against an IndicVoices Hindi sample."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Prevent an inherited shell environment from accidentally using a metered API.
os.environ["SPEECHMIND_ASR_PROVIDER"] = "local"

from models.asr import SpeechRecognizer
from scripts.asr_metrics import aggregate_error_counts, error_counts
from utils.audio_utils import load_audio


DEFAULT_SAMPLE_DIR = Path("data/indicvoices/hindi/valid_sample")


def read_manifest(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def transcribe_one(recognizer: SpeechRecognizer, audio, model: str) -> tuple[str, str]:
    if model == "auto":
        result = recognizer.transcribe(audio)
        return str(result.get("transcript") or ""), str(result.get("asr_model") or "auto")
    if model == "whisper":
        raw = recognizer._decode_with_whisper(audio, language=None)
        result = recognizer._whisper_result(raw, language_source="benchmark_whisper_auto")
        return str(result.get("transcript") or ""), "whisper-small"
    if model == "indicconformer":
        return recognizer._indic_transcribe(audio, "hi"), "indic-conformer-600m"
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
        recognizer = SpeechRecognizer()

        rows = []
        normalized_metrics = []
        verbatim_metrics = []
        scenario_groups = defaultdict(list)
        total_started = time.perf_counter()

        for index, record in enumerate(records, start=1):
            audio_path = args.sample_dir / record["audio_file"]
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing audio file: {audio_path}")

            audio = load_audio(str(audio_path))
            started = time.perf_counter()
            hypothesis, actual_model = transcribe_one(recognizer, audio, args.model)
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

        elapsed = time.perf_counter() - total_started
        summary = {
            "dataset": "ai4bharat/IndicVoices",
            "config": "hindi",
            "split": "valid",
            "model_requested": args.model,
            "samples": len(rows),
            "unique_speakers": len({row["speaker_id"] for row in rows}),
            "normalized_reference_metrics": aggregate_error_counts(normalized_metrics),
            "verbatim_reference_metrics": aggregate_error_counts(verbatim_metrics),
            "by_scenario": {
                scenario: aggregate_error_counts(group)
                for scenario, group in scenario_groups.items()
            },
            "mean_inference_seconds": sum(
                row["inference_seconds"] for row in rows
            ) / max(1, len(rows)),
            "total_elapsed_seconds": round(elapsed, 3),
            "predictions_csv": str(csv_path),
            "notes": [
                "WER and CER are computed after NFKC normalization, lowercasing, punctuation removal, and whitespace collapse.",
                "Normalized and verbatim references are reported separately.",
                "This is ASR evaluation only; IndicVoices has no emotion labels in this manifest.",
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
