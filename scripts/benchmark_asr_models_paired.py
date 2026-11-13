"""Paired ASR comparison on the exact same HiACC recordings.

Runs Srota with automatic vocabulary context, Srota without context,
IndicConformer forced to Hindi, and Whisper with automatic language detection.
All candidates use identical audio paths, references, and normalization.
The production-router transcript already present in the input CSV is retained
as a historical baseline, not treated as another independent model run.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
import random
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.asr_metrics import aggregate_error_counts, error_counts_multi_reference
from scripts.asr_error_analysis import build_word_error_profile

DATA_ROOT = PROJECT_ROOT / "data" / "hiacc"
DEFAULT_OUTPUT = DATA_ROOT / "benchmarks" / "paired_asr"


def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Paired benchmark input not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    if not rows:
        raise ValueError(f"No rows found in {path}")

    audio_key = "audio_path" if "audio_path" in columns else None
    reference_key = (
        "reference_transcript" if "reference_transcript" in columns
        else "reference" if "reference" in columns else None
    )
    if audio_key is None or reference_key is None:
        raise ValueError(
            "Input CSV must contain audio_path and either reference_transcript "
            "or reference columns."
        )

    normalized = []
    seen_paths = set()
    for index, raw in enumerate(rows, start=2):
        row = {str(key): str(value or "").strip() for key, value in raw.items()}
        audio_path = row.get(audio_key, "").replace("\\", "/")
        reference = row.get(reference_key, "").strip()
        speaker = row.get("speaker_id", "").strip()
        if not audio_path or not reference:
            raise ValueError(
                f"Input row {index} needs non-empty audio_path and reference transcript."
            )
        if not speaker:
            raise ValueError(
                f"Input row {index} has no speaker_id. Refusing to claim speaker-aware "
                "sampling without trustworthy speaker identifiers."
            )
        key = audio_path.casefold()
        if key in seen_paths:
            raise ValueError(
                f"Duplicate audio_path in input row {index}: {audio_path}"
            )
        seen_paths.add(key)
        row["audio_path"] = audio_path
        row["reference_transcript"] = reference
        row["speaker_id"] = speaker
        row["age_group"] = row.get("age_group") or "unknown"
        dataset_value = row.get("dataset") or "hiacc"
        row["dataset"] = (
            "hiacc" if dataset_value.casefold() in {"unspecified", "unknown"}
            else dataset_value
        )
        row["production_hypothesis"] = (
            row.get("hypothesis_transcript")
            or row.get("hypothesis")
            or ""
        )
        normalized.append(row)
    return normalized


def select_records(
    records: list[dict],
    limit: int = 20,
    seed: int = 42,
    max_per_speaker: int = 1,
) -> list[dict]:
    """Deterministic approximately age-balanced sampling with a speaker cap."""
    if limit < 0:
        raise ValueError("limit must be 0 (all records) or a positive integer.")
    if max_per_speaker < 1:
        raise ValueError("max_per_speaker must be at least 1.")

    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[row.get("age_group") or "unknown"].append(row)
    for values in groups.values():
        rng.shuffle(values)

    keys = sorted(groups, key=lambda value: (value != "adult", value != "children", value))
    speaker_counts: dict[tuple[str, str], int] = defaultdict(int)
    selected = []
    while keys and (limit == 0 or len(selected) < limit):
        remaining = []
        added = False
        for key in keys:
            pool = groups[key]
            chosen = None
            while pool:
                candidate = pool.pop()
                speaker_key = (
                    candidate.get("dataset") or "hiacc",
                    candidate["speaker_id"],
                )
                if speaker_counts[speaker_key] >= max_per_speaker:
                    continue
                chosen = candidate
                speaker_counts[speaker_key] += 1
                break
            if chosen is not None:
                selected.append(chosen)
                added = True
            if pool:
                remaining.append(key)
            if limit and len(selected) >= limit:
                break
        keys = remaining
        if not added:
            break
    return selected


def resolve_audio_path(value: str, audio_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (audio_root / path).resolve()


def extract_whisper_text(result) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("text") or result.get("transcript") or "").strip()
    return str(result).strip()


def script_class(text: str) -> str:
    scripts = set()
    for char in str(text or ""):
        name = unicodedata.name(char, "")
        if "DEVANAGARI" in name:
            scripts.add("Devanagari")
        elif "LATIN" in name:
            scripts.add("Latin")
        elif "ARABIC" in name:
            scripts.add("Arabic")
    if not scripts:
        return "empty_or_other"
    return "+".join(sorted(scripts))


def optional_median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def default_run_output_dir(now: datetime | None = None) -> Path:
    """Return a unique UTC-stamped directory so earlier results are preserved."""
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT / f"run-{timestamp}"


def status_key_for_hypothesis(hypothesis_key: str) -> str:
    suffix = "_hypothesis"
    if not hypothesis_key.endswith(suffix):
        raise ValueError(f"Unexpected hypothesis column: {hypothesis_key}")
    return hypothesis_key[:-len(suffix)] + "_status"


def seconds_key_for_hypothesis(hypothesis_key: str) -> str:
    suffix = "_hypothesis"
    if not hypothesis_key.endswith(suffix):
        raise ValueError(f"Unexpected hypothesis column: {hypothesis_key}")
    return hypothesis_key[:-len(suffix)] + "_seconds"


def aggregate_predictions(rows: list[dict], hypothesis_key: str) -> dict:
    status_key = status_key_for_hypothesis(hypothesis_key)
    seconds_key = seconds_key_for_hypothesis(hypothesis_key)
    all_selected_scores = [
        error_counts_multi_reference(
            row["reference_transcript"],
            str(row.get(hypothesis_key) or ""),
        )
        for row in rows
    ]
    successful_scores = [
        error_counts_multi_reference(
            row["reference_transcript"],
            str(row.get(hypothesis_key) or ""),
        )
        for row in rows if row.get(status_key) == "ok"
    ]
    times = [
        float(row[seconds_key])
        for row in rows
        if row.get(status_key) == "ok"
        and row.get(seconds_key) not in (None, "")
    ]
    return {
        "selected_samples": len(rows),
        "successful_samples": len(successful_scores),
        "failed_samples": len(rows) - len(successful_scores),
        "micro": (
            aggregate_error_counts(all_selected_scores)
            if all_selected_scores else None
        ),
        "micro_successful_samples_only": (
            aggregate_error_counts(successful_scores)
            if successful_scores else None
        ),
        "median_inference_seconds": optional_median(times),
        "failure_policy": (
            "Failed model calls are scored as empty hypotheses in micro WER/CER; "
            "successful-only metrics are also provided for diagnosis."
        ),
    }


def grouped_metrics(rows: list[dict], hypothesis_key: str, group_key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key) or "unknown")].append(row)
    output = {}
    for group, values in sorted(groups.items()):
        scored = [
            error_counts_multi_reference(
                row["reference_transcript"], str(row.get(hypothesis_key) or "")
            )
            for row in values
        ]
        output[group] = aggregate_error_counts(scored) if scored else None
    return output


def paired_vs_production(rows: list[dict], hypothesis_key: str) -> dict | None:
    comparable = []
    for row in rows:
        if row.get("production_hypothesis"):
            candidate = error_counts_multi_reference(
                row["reference_transcript"], str(row.get(hypothesis_key) or "")
            )
            production = error_counts_multi_reference(
                row["reference_transcript"], row["production_hypothesis"]
            )
            comparable.append((candidate["wer"], production["wer"]))
    if not comparable:
        return None
    improved = sum(candidate < production for candidate, production in comparable)
    worsened = sum(candidate > production for candidate, production in comparable)
    tied = len(comparable) - improved - worsened
    return {
        "paired_samples": len(comparable),
        "candidate_better": improved,
        "candidate_worse": worsened,
        "tied": tied,
        "mean_clip_wer_delta_vs_production": round(
            statistics.mean(candidate - production for candidate, production in comparable), 4
        ),
        "note": "Negative WER delta favours the candidate; this is paired per-clip comparison.",
    }


def paired_model_comparison(
    rows: list[dict], candidate_key: str, reference_key: str
) -> dict | None:
    """Compare two hypotheses on identical rows, counting failures as empty text."""
    if not rows:
        return None
    deltas = []
    for row in rows:
        candidate = error_counts_multi_reference(
            row["reference_transcript"], str(row.get(candidate_key) or "")
        )
        reference = error_counts_multi_reference(
            row["reference_transcript"], str(row.get(reference_key) or "")
        )
        deltas.append((candidate["wer"], reference["wer"]))
    improved = sum(candidate < reference for candidate, reference in deltas)
    worsened = sum(candidate > reference for candidate, reference in deltas)
    tied = len(deltas) - improved - worsened
    return {
        "paired_samples": len(deltas),
        "candidate_better": improved,
        "candidate_worse": worsened,
        "tied": tied,
        "mean_clip_wer_delta": round(
            statistics.mean(candidate - reference for candidate, reference in deltas), 4
        ),
        "note": "Negative delta favours the candidate; failed hypotheses are scored as empty.",
    }


def _run_srota_call(provider, audio, context: str | None) -> dict:
    started = time.perf_counter()
    try:
        result = provider.transcribe_audio(
            audio, sample_rate=16000, context=context
        )
        hypothesis = str(result.get("transcript") or "").strip()
        if not hypothesis:
            raise RuntimeError("Srota returned an empty transcript.")
        return {
            "hypothesis": hypothesis,
            "status": "ok",
            "seconds": round(time.perf_counter() - started, 3),
            "error": "",
        }
    except Exception as error:
        return {
            "hypothesis": "",
            "status": "error",
            "seconds": "",
            "error": f"{type(error).__name__}: {error}",
        }


def run_srota_variants(provider, audio, context: str | None) -> tuple[dict, bool]:
    """Run two true variants only when the context string is non-empty."""
    auto_result = _run_srota_call(provider, audio, context or None)
    shared_no_context_call = not bool(context)
    native_result = (
        dict(auto_result)
        if shared_no_context_call
        else _run_srota_call(provider, audio, None)
    )
    variants = {}
    for key, result in (
        ("srota_auto", auto_result),
        ("srota_native", native_result),
    ):
        variants[f"{key}_hypothesis"] = result["hypothesis"]
        variants[f"{key}_status"] = result["status"]
        variants[f"{key}_seconds"] = result["seconds"]
        if result["error"]:
            variants[f"{key}_error"] = result["error"]
    variants["srota_native_shared_call"] = shared_no_context_call
    return variants, shared_no_context_call


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Full-pipeline predictions.csv or a HiACC benchmark predictions.csv. "
             "When omitted, the newest data/benchmarks/full_pipeline/run-*/predictions.csv is used.",
    )
    parser.add_argument("--audio-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: data/hiacc/benchmarks/paired_asr/run-<UTC timestamp>.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-speaker", type=int, default=1)
    parser.add_argument("--indic-language", default="hi")
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--skip-indicconformer", action="store_true")
    parser.add_argument("--skip-srota", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = default_run_output_dir()

    started = time.perf_counter()
    recognizer = None
    srota_provider = None
    try:
        input_path = args.input
        if input_path is None:
            candidates = sorted(
                (PROJECT_ROOT / "data" / "benchmarks" / "full_pipeline").glob(
                    "run-*/predictions.csv"
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise FileNotFoundError(
                    "No full-pipeline evaluation predictions.csv found. "
                    "Run EVALUATE_SPEECHMIND_HIACC.bat first or pass --input."
                )
            input_path = candidates[0]

        records = read_records(input_path)
        selected = select_records(
            records, limit=args.limit, seed=args.seed,
            max_per_speaker=args.max_per_speaker,
        )
        if not selected:
            raise RuntimeError("No records selected for paired ASR comparison.")

        for row in selected:
            audio_path = resolve_audio_path(row["audio_path"], args.audio_root)
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing audio file: {audio_path}")
            row["_resolved_audio_path"] = str(audio_path)

        print(
            f"[PairedASR] Input: {input_path}\n"
            f"[PairedASR] Same recordings for every candidate: {len(selected)}\n"
            f"[PairedASR] Speakers: {len({row['speaker_id'] for row in selected})}\n"
            "[PairedASR] Candidate models: Srota auto-context, Srota no-context, "
            "IndicConformer forced language, Whisper auto-language."
        )

        import torch

        torch.set_num_threads(4)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass

        from models.asr import SpeechRecognizer
        from models.srota_provider import SrotaASRProvider
        from utils.audio_utils import load_audio

        init_started = time.perf_counter()
        recognizer = SpeechRecognizer()
        whisper_initialization_seconds = time.perf_counter() - init_started

        indic_initialization_seconds = None
        if not args.skip_indicconformer:
            indic_started = time.perf_counter()
            recognizer._load_indic_model()
            indic_initialization_seconds = time.perf_counter() - indic_started

        srota_startup_and_warmup_seconds = None
        if not args.skip_srota:
            srota_provider = SrotaASRProvider(project_root=PROJECT_ROOT)
            if not srota_provider.available:
                raise FileNotFoundError(
                    "Srota worker environment is not installed. Run "
                    "SETUP_HINGLISH_ASR_ENV.bat before this benchmark."
                )
            warmup_audio = load_audio(selected[0]["_resolved_audio_path"])
            warmup_started = time.perf_counter()
            warmup_result = srota_provider.transcribe_audio(
                warmup_audio, sample_rate=16000, context=None
            )
            if not str(warmup_result.get("transcript") or "").strip():
                raise RuntimeError("Srota warm-up returned an empty transcript.")
            srota_startup_and_warmup_seconds = time.perf_counter() - warmup_started
            print(
                "[PairedASR] Srota worker/model startup + warm-up: "
                f"{srota_startup_and_warmup_seconds:.1f}s (not scored)"
            )

        output_rows = []
        for index, row in enumerate(selected, start=1):
            audio_path = Path(row["_resolved_audio_path"])
            audio = load_audio(str(audio_path))
            output = {
                "audio_path": row["audio_path"],
                "audio_filename": audio_path.name,
                "speaker_id": row["speaker_id"],
                "age_group": row["age_group"],
                "dataset": row["dataset"],
                "reference_transcript": row["reference_transcript"],
                "reference_script": script_class(row["reference_transcript"]),
                "production_model": row.get("asr_model", ""),
                "production_hypothesis": row.get("production_hypothesis", ""),
            }

            if not args.skip_srota:
                (
                    context,
                    context_source,
                    probe_seconds,
                    probe_transcript,
                    probe_language,
                    probe_language_score,
                ) = recognizer._infer_srota_context(audio)
                output["srota_context_terms"] = context
                output["srota_context_source"] = context_source
                output["srota_context_probe_seconds"] = round(float(probe_seconds or 0.0), 3)
                output["srota_context_probe_transcript"] = probe_transcript
                output["srota_context_probe_language"] = probe_language
                output["srota_context_probe_language_score"] = probe_language_score

                variant_results, shared_no_context_call = run_srota_variants(
                    srota_provider, audio, context
                )
                output.update(variant_results)
                output["srota_auto_total_seconds"] = (
                    round(
                        float(output["srota_auto_seconds"] or 0.0)
                        + float(probe_seconds or 0.0),
                        3,
                    )
                    if output["srota_auto_status"] == "ok" else ""
                )

            if not args.skip_indicconformer:
                call_started = time.perf_counter()
                try:
                    hypothesis = recognizer._indic_transcribe(audio, args.indic_language)
                    output["indicconformer_hypothesis"] = hypothesis
                    output["indicconformer_status"] = "ok"
                    output["indicconformer_seconds"] = round(
                        time.perf_counter() - call_started, 3
                    )
                except Exception as error:
                    output["indicconformer_hypothesis"] = ""
                    output["indicconformer_status"] = "error"
                    output["indicconformer_seconds"] = ""
                    output["indicconformer_error"] = f"{type(error).__name__}: {error}"

            if not args.skip_whisper:
                call_started = time.perf_counter()
                try:
                    result = recognizer._decode_with_whisper(audio, language=None)
                    hypothesis = extract_whisper_text(result)
                    if not hypothesis:
                        raise RuntimeError("Whisper returned an empty transcript.")
                    output["whisper_auto_hypothesis"] = hypothesis
                    output["whisper_auto_status"] = "ok"
                    output["whisper_auto_seconds"] = round(
                        time.perf_counter() - call_started, 3
                    )
                    output["whisper_detected_language"] = str(
                        result.get("language") if isinstance(result, dict) else ""
                    )
                except Exception as error:
                    output["whisper_auto_hypothesis"] = ""
                    output["whisper_auto_status"] = "error"
                    output["whisper_auto_seconds"] = ""
                    output["whisper_auto_error"] = f"{type(error).__name__}: {error}"

            output_rows.append(output)
            print(f"[PairedASR] {index}/{len(selected)} completed: {audio_path.name}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = args.output_dir / "predictions.csv"
        write_predictions(predictions_path, output_rows)

        model_columns = {
            "srota_auto": "srota_auto_hypothesis",
            "srota_native": "srota_native_hypothesis",
            "indicconformer": "indicconformer_hypothesis",
            "whisper_auto": "whisper_auto_hypothesis",
        }
        model_results = {}
        for model_name, hypothesis_key in model_columns.items():
            status_key = f"{model_name}_status"
            if all(status_key not in row for row in output_rows):
                continue
            model_results[model_name] = {
                **aggregate_predictions(output_rows, hypothesis_key),
                "by_age_group": grouped_metrics(output_rows, hypothesis_key, "age_group"),
                "by_reference_script": grouped_metrics(
                    output_rows, hypothesis_key, "reference_script"
                ),
                "paired_vs_production_router": paired_vs_production(
                    output_rows, hypothesis_key
                ),
                "word_error_profile": build_word_error_profile(
                    output_rows, hypothesis_key, status_key, limit=20
                ),
            }
        if "srota_auto" in model_results:
            effective_times = [
                float(row["srota_auto_total_seconds"])
                for row in output_rows
                if row.get("srota_auto_status") == "ok"
                and row.get("srota_auto_total_seconds") not in (None, "")
            ]
            probe_times = [
                float(row["srota_context_probe_seconds"])
                for row in output_rows
                if row.get("srota_auto_status") == "ok"
            ]
            model_results["srota_auto"]["median_effective_seconds_including_context_scout"] = (
                optional_median(effective_times)
            )
            model_results["srota_auto"]["median_context_scout_seconds"] = (
                optional_median(probe_times)
            )
            model_results["srota_auto"]["paired_vs_srota_native"] = (
                paired_model_comparison(
                    output_rows,
                    "srota_auto_hypothesis",
                    "srota_native_hypothesis",
                )
                if "srota_native" in model_results else None
            )

        summary = {
            "benchmark": "SpeechMind paired direct-model ASR comparison",
            "input_predictions_csv": str(input_path.resolve()),
            "dataset": "HiACC" if all(row["dataset"] == "hiacc" for row in selected) else "mixed_or_custom",
            "split": sorted({row.get("split", "unknown") for row in selected}),
            "samples": len(selected),
            "unique_speakers": len({row["speaker_id"] for row in selected}),
            "selection": {
                "seed": args.seed,
                "limit": args.limit,
                "max_per_speaker": args.max_per_speaker,
                "age_group_counts": {
                    group: sum(row["age_group"] == group for row in selected)
                    for group in sorted({row["age_group"] for row in selected})
                },
                "note": (
                    "Every direct candidate receives the exact same selected audio files "
                    "and references. This paired comparison does not remove pretraining "
                    "or fine-tuning exposure to speakers in the dataset."
                ),
            },
            "initialization_seconds": {
                "whisper_and_recognizer": round(whisper_initialization_seconds, 3),
                "indicconformer": (
                    round(indic_initialization_seconds, 3)
                    if indic_initialization_seconds is not None else None
                ),
                "srota_worker_and_model_startup_plus_warmup_seconds": (
                    round(srota_startup_and_warmup_seconds, 3)
                    if srota_startup_and_warmup_seconds is not None else None
                ),
            },
            "models": model_results,
            "production_router_reference": {
                "note": (
                    "Historical output from the input CSV, not re-run as part of this "
                    "paired benchmark. It may contain automatic routing and context."
                ),
                "micro": aggregate_error_counts([
                    error_counts_multi_reference(
                        row["reference_transcript"], row["production_hypothesis"]
                    )
                    for row in output_rows if row["production_hypothesis"]
                ]) if any(row["production_hypothesis"] for row in output_rows) else None,
            },
            "normalization": (
                "SpeechMind NFKC + lowercase + punctuation removal; "
                "alternative-aware references; micro corpus WER/CER."
            ),
            "benchmark_elapsed_seconds": round(time.perf_counter() - started, 3),
            "predictions_csv": str(predictions_path.resolve()),
            "notes": [
                "Srota auto-context mirrors production's Whisper tail vocabulary scout.",
                "Srota native disables vocabulary context to measure the context contribution.",
                "The Srota worker/model is warmed on the first selected clip before any scored runs; the warm-up is timed separately and excluded from ASR scores.",
                "Srota auto-context effective per-clip latency includes both the Whisper context scout and Srota inference; both components are retained separately in predictions.csv.",
                "When the scout returns an empty context, Srota is called only once and the identical no-context result is reused in both variant columns; srota_native_shared_call marks this case.",
                "IndicConformer is forced to the supplied Hindi code; this is a diagnostic on Hinglish, not a claim that it is a Hinglish specialist.",
                "Whisper uses automatic language detection and transcription.",
                "All candidates are scored against identical references selected once at the start.",
                "Do not change production routing based on this small diagnostic sample alone.",
            ],
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\n[PairedASR] Benchmark complete.")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"\n[PairedASR] ERROR: {type(error).__name__}: {error}")
        return 1
    finally:
        if srota_provider is not None:
            srota_provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
