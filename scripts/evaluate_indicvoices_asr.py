"""Benchmark local SpeechMind ASR against a bounded IndicVoices Hindi sample."""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# Make direct execution from scripts/ resolve top-level project packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The benchmark is local-only; never inherit a metered provider from the shell.
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


def detect_script(text: str) -> str:
    """Coarsely summarize Unicode scripts in ASR output (not language ID)."""
    scripts = set()
    for char in str(text or ""):
        name = unicodedata.name(char, "")
        if "DEVANAGARI" in name:
            scripts.add("Devanagari")
        elif "ARABIC" in name:
            scripts.add("Arabic")
        elif "LATIN" in name:
            scripts.add("Latin")
        elif "HIRAGANA" in name or "KATAKANA" in name or "CJK" in name:
            scripts.add("CJK/Japanese")
        elif "HANGUL" in name:
            scripts.add("Hangul")
    return "+".join(sorted(scripts)) if scripts else "empty_or_other"


def duration_bucket(duration: float) -> str:
    if duration < 1.0:
        return "<1s"
    if duration < 1.5:
        return "1-1.5s"
    if duration < 3.0:
        return "1.5-3s"
    return ">=3s"


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
            result.get("language"),
        )
    if model in ("whisper", "whisper-hi"):
        # "whisper" measures the model's own language identification;
        # "whisper-hi" is a diagnostic forced-Hindi comparison. Neither
        # option changes SpeechMind's no-language-selector product UX.
        language = "hi" if model == "whisper-hi" else None
        raw = recognizer._decode_with_whisper(audio, language=language)
        source = (
            "benchmark_whisper_forced_hindi"
            if language == "hi" else "benchmark_whisper_auto"
        )
        result = recognizer._whisper_result(raw, language_source=source)
        output_language = result.get("language")
        if language == "hi":
            print(
                "[ASR Benchmark] Forced language=hi; "
                f"Whisper result language={output_language!r}"
            )
            if output_language and str(output_language).lower() != "hi":
                raise RuntimeError(
                    "Forced-Hindi benchmark verification failed: Whisper returned "
                    f"language={output_language!r}, expected 'hi'. Stopping after "
                    "this clip to avoid wasting time. Check that the forced language "
                    "argument reaches the installed Whisper decoder."
                )
        model_label = (
            "whisper-small-forced-hi" if language == "hi" else "whisper-small-auto"
        )
        return (
            str(result.get("transcript") or ""),
            model_label,
            False,
            None,
            output_language,
        )
    if model == "indicconformer":
        result = recognizer._indic_result_with_whisper_fallback(audio, "hi")
        return (
            str(result.get("transcript") or ""),
            str(result.get("asr_model") or "indic-conformer-600m"),
            bool(result.get("asr_fallback", False)),
            result.get("asr_fallback_reason"),
            result.get("language"),
        )
    raise ValueError(f"Unknown ASR model: {model}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument(
        "--model", choices=("auto", "whisper", "whisper-hi", "indicconformer"), default="auto"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Optional maximum number of records; 0 evaluates the complete sample.",
    )
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    manifest_path = sample_dir / "manifest.jsonl"
    total_started = time.perf_counter()
    try:
        records = read_manifest(manifest_path)
        if args.limit > 0:
            records = records[:args.limit]
        if not records:
            raise ValueError("No benchmark records selected.")

        print(f"[ASR Benchmark] Model={args.model}; clips={len(records)}")
        print("[ASR Benchmark] Local inference only; no API provider is used.")

        initialization_started = time.perf_counter()
        recognizer = SpeechRecognizer()

        indicconformer_warmup_error = None
        if args.model in ("auto", "indicconformer"):
            print("[ASR Benchmark] Initializing IndicConformer before clip timing...")
            try:
                recognizer._load_indic_model()
            except Exception as error:
                indicconformer_warmup_error = f"{type(error).__name__}: {error}"
                print(
                    "[ASR Benchmark] IndicConformer initialization failed; "
                    "the recognizer's Whisper fallback will be measured."
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
        duration_groups = defaultdict(list)
        model_groups = defaultdict(list)
        script_counts = Counter()

        for index, record in enumerate(records, start=1):
            audio_path = sample_dir / record["audio_file"]
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing audio file: {audio_path}")

            audio = load_audio(str(audio_path))
            started = time.perf_counter()
            (
                hypothesis,
                actual_model,
                used_fallback,
                fallback_reason,
                decoder_language,
            ) = transcribe_one(recognizer, audio, args.model)
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
            duration = float(record.get("duration") or 0.0)
            output_script = detect_script(hypothesis)
            scenario_groups[scenario].append(normalized_result)
            duration_groups[duration_bucket(duration)].append(normalized_result)
            model_groups[actual_model].append(normalized_result)
            script_counts[output_script] += 1

            rows.append({
                "sample_id": record.get("sample_id"),
                "speaker_id": record.get("speaker_id"),
                "scenario": scenario,
                "duration_seconds": duration,
                "model_requested": args.model,
                "model_used": actual_model,
                "decoder_language": decoder_language,
                "hypothesis_script": output_script,
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
                current_wer = aggregate_error_counts(normalized_metrics)["wer"]
                print(
                    f"[ASR Benchmark] {index}/{len(records)} complete; "
                    f"last={inference_seconds:.1f}s; WER so far={current_wer:.3f}"
                )

        clip_elapsed = time.perf_counter() - inference_started
        elapsed = time.perf_counter() - total_started
        inference_values = [row["inference_seconds"] for row in rows]

        output_dir = sample_dir / "benchmarks" / args.model
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "predictions.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "dataset": "ai4bharat/IndicVoices",
            "config": "hindi",
            "split": "valid",
            "model_requested": args.model,
            "samples": len(rows),
            "fallback_count": sum(1 for row in rows if row["asr_fallback"]),
            "unique_speakers": len({
                row["speaker_id"] for row in rows if row["speaker_id"]
            }),
            "models_used": {
                model: {
                    "samples": len(group),
                    "metrics": aggregate_error_counts(group),
                }
                for model, group in model_groups.items()
            },
            "hypothesis_script_counts": dict(script_counts),
            "normalized_reference_metrics": aggregate_error_counts(normalized_metrics),
            "verbatim_reference_metrics": aggregate_error_counts(verbatim_metrics),
            "by_scenario": {
                key: aggregate_error_counts(group)
                for key, group in scenario_groups.items()
            },
            "by_duration": {
                key: aggregate_error_counts(group)
                for key, group in duration_groups.items()
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
                "Model initialization is reported separately from per-clip inference.",
                "Output script breakdown is a Unicode-script diagnostic, not language identification.",
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
