"""Evaluate the production SpeechMind pipeline on a speaker-aware CSV manifest.

This runner scores ASR against reference transcripts, optionally scores English
meaning against human reference translations, and optionally measures text/voice
emotion classification when gold emotion labels are supplied. It does not infer
gold labels or treat an unavailable branch as a correct prediction.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.asr_metrics import aggregate_error_counts, error_counts_multi_reference


REQUIRED_COLUMNS = {"audio_path", "transcript", "speaker_id", "split"}
OPTIONAL_COLUMNS = {
    "dataset", "age_group", "language", "reference_meaning_en", "emotion_label"
}
EMOTION_ALIASES = {
    "happy": "joy",
    "happiness": "joy",
    "sad": "sadness",
    "fearful": "fear",
    "angry": "anger",
    "disgusted": "disgust",
    "surprised": "surprise",
}
ABSTAIN_LABELS = {"", "uncertain", "unavailable", "unknown", "abstain"}


def read_manifest(path: Path, split: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return all normalized rows and the requested evaluation split."""
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}. "
                "See docs/MULTI_RECORDING_EVALUATION.md for the schema."
            )
        all_rows = []
        for line_number, raw in enumerate(reader, start=2):
            row = {str(key): str(value or "").strip() for key, value in raw.items()}
            row["_line_number"] = str(line_number)
            row["dataset"] = row.get("dataset") or path.parent.name or "unspecified"
            if not row.get("language"):
                row["language"] = (
                    "hinglish" if "hiacc" in row["dataset"].casefold() else "unknown"
                )
            row["speaker_id"] = row["speaker_id"].strip()
            row["split"] = row["split"].strip().lower()
            if not row["audio_path"] or not row["transcript"]:
                raise ValueError(
                    f"Manifest row {line_number} requires non-empty audio_path and transcript."
                )
            if not row["speaker_id"]:
                raise ValueError(
                    f"Manifest row {line_number} has no speaker_id. "
                    "Speaker-aware sampling cannot be performed safely."
                )
            all_rows.append(row)

    requested = [row for row in all_rows if row["split"] == split.lower()]
    if not requested:
        available = sorted({row["split"] for row in all_rows})
        raise ValueError(
            f"No rows found for split={split!r}; available splits: {available}"
        )
    return all_rows, requested


def speaker_split_report(all_rows: list[dict[str, str]], split: str) -> dict:
    """Measure speaker overlap between evaluation and train/validation splits."""
    evaluation = [row for row in all_rows if row["split"] == split.lower()]
    reference_splits = {"train", "training", "validation", "valid", "dev"}
    training_rows = [row for row in all_rows if row["split"] in reference_splits]

    eval_by_dataset: dict[str, set[str]] = defaultdict(set)
    train_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in evaluation:
        eval_by_dataset[row["dataset"]].add(row["speaker_id"])
    for row in training_rows:
        train_by_dataset[row["dataset"]].add(row["speaker_id"])

    overlap = {
        dataset: sorted(eval_speakers & train_by_dataset.get(dataset, set()))
        for dataset, eval_speakers in eval_by_dataset.items()
    }
    overlap = {key: value for key, value in overlap.items() if value}
    evaluation_speakers = {
        (row["dataset"], row["speaker_id"]) for row in evaluation
    }
    training_speakers = {
        (row["dataset"], row["speaker_id"]) for row in training_rows
    }
    return {
        "evaluation_split": split,
        "training_validation_rows_in_manifest": len(training_rows),
        "evaluation_speaker_count": len(evaluation_speakers),
        "overlapping_speaker_count": sum(len(value) for value in overlap.values()),
        "overlapping_speakers_by_dataset": overlap,
        "unseen_evaluation_speaker_count": len(evaluation_speakers - training_speakers),
        "speaker_disjoint_from_manifest_train_validation": (
            bool(training_rows) and not overlap
        ),
        "status": (
            "speaker_disjoint"
            if training_rows and not overlap
            else "speaker_overlap"
            if overlap
            else "not_assessable_no_train_validation_rows"
        ),
    }


def unseen_speaker_records(
    all_rows: list[dict[str, str]], evaluation_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Keep only evaluation speakers absent from train/validation/dev rows."""
    train_splits = {"train", "training", "validation", "valid", "dev"}
    training_speakers = {
        (row["dataset"], row["speaker_id"])
        for row in all_rows if row["split"] in train_splits
    }
    return [
        row for row in evaluation_rows
        if (row["dataset"], row["speaker_id"]) not in training_speakers
    ]


def select_records(
    records: list[dict[str, str]],
    limit: int,
    seed: int = 42,
    max_per_speaker: int = 1,
) -> list[dict[str, str]]:
    """Select reproducible, approximately group-balanced clips with a speaker cap."""
    if limit < 0:
        raise ValueError("limit must be zero (all eligible records) or a positive integer.")
    if max_per_speaker < 1:
        raise ValueError("max_per_speaker must be at least 1.")

    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        stratum = (
            row.get("age_group") or row.get("language") or "all"
        ).strip().lower()
        groups[(row.get("dataset") or "unspecified", stratum)].append(row)

    for rows in groups.values():
        rng.shuffle(rows)

    group_keys = sorted(groups)
    selected: list[dict[str, str]] = []
    speaker_counts: dict[tuple[str, str], int] = {}
    while group_keys and (limit == 0 or len(selected) < limit):
        remaining = []
        for key in group_keys:
            pool = groups[key]
            chosen = None
            while pool:
                candidate = pool.pop()
                speaker_key = (candidate["dataset"], candidate["speaker_id"])
                if speaker_counts.get(speaker_key, 0) >= max_per_speaker:
                    continue
                chosen = candidate
                speaker_counts[speaker_key] = speaker_counts.get(speaker_key, 0) + 1
                break
            if chosen is not None:
                selected.append(chosen)
            if pool:
                remaining.append(key)
            if limit and len(selected) >= limit:
                break
        group_keys = remaining
    return selected


def resolve_audio_path(audio_path: str, audio_root: Path) -> Path:
    path = Path(audio_path)
    return path.resolve() if path.is_absolute() else (audio_root / path).resolve()


def canonical_emotion(label: Any) -> str:
    value = str(label or "").strip().lower()
    return EMOTION_ALIASES.get(value, value)


def classification_metrics(rows: list[dict], prediction_key: str) -> dict | None:
    """Report coverage, covered accuracy, all-row accuracy and macro-F1."""
    labelled = [
        row for row in rows
        if str(row.get("emotion_label") or "").strip()
    ]
    if not labelled:
        return None

    normalized = [
        (
            canonical_emotion(row.get("emotion_label")),
            canonical_emotion(row.get(prediction_key)),
        )
        for row in labelled
    ]
    covered = [(truth, pred) for truth, pred in normalized if pred not in ABSTAIN_LABELS]
    correct_covered = sum(truth == pred for truth, pred in covered)
    correct_all = sum(truth == pred for truth, pred in normalized)
    labels = sorted({label for pair in covered for label in pair})
    f1_values = []
    for label in labels:
        tp = sum(truth == label and pred == label for truth, pred in covered)
        fp = sum(truth != label and pred == label for truth, pred in covered)
        fn = sum(truth == label and pred != label for truth, pred in covered)
        denominator = (2 * tp) + fp + fn
        f1_values.append((2 * tp / denominator) if denominator else 0.0)

    total = len(labelled)
    return {
        "labelled_samples": total,
        "covered_samples": len(covered),
        "abstained_samples": total - len(covered),
        "coverage": len(covered) / total,
        "accuracy_all_labelled_samples": correct_all / total,
        "accuracy_on_covered_samples": (
            correct_covered / len(covered) if covered else None
        ),
        "macro_f1_on_covered_samples": (
            statistics.mean(f1_values) if f1_values else None
        ),
        "labels_in_covered_sample": labels,
        "note": (
            "Coverage and accuracy are separate. Uncertain/unavailable outputs "
            "count as abstentions and are incorrect for all-sample accuracy."
        ),
    }


def grouped_asr_metrics(rows: list[dict], group_key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key) or "unknown")].append(row["asr_metrics"])
    return {
        group: aggregate_error_counts(metrics)
        for group, metrics in sorted(groups.items())
    }


def grouped_timing_medians(rows: list[dict]) -> dict:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for stage, seconds in (row.get("timings_sec") or {}).items():
            try:
                values[stage].append(float(seconds))
            except (TypeError, ValueError):
                continue
    return {
        stage: {
            "median_seconds": round(statistics.median(samples), 3),
            "mean_seconds": round(statistics.mean(samples), 3),
            "samples": len(samples),
        }
        for stage, samples in sorted(values.items())
        if samples
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def current_git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_evaluation(
    manifest_path: Path,
    audio_root: Path,
    output_dir: Path,
    split: str = "test",
    limit: int = 40,
    seed: int = 42,
    max_per_speaker: int = 1,
    strict_speaker_holdout: bool = False,
) -> dict:
    all_rows, evaluation_rows = read_manifest(manifest_path, split)
    split_report = speaker_split_report(all_rows, split)
    eligible_rows = list(evaluation_rows)
    if strict_speaker_holdout:
        if not split_report["training_validation_rows_in_manifest"]:
            raise RuntimeError(
                "Strict speaker holdout cannot be checked because the manifest "
                "contains no train/validation/dev rows. Do not claim an unseen-speaker result."
            )
        eligible_rows = unseen_speaker_records(all_rows, evaluation_rows)
        if not eligible_rows:
            raise RuntimeError(
                "Strict speaker holdout selected zero evaluation rows: every speaker "
                "in this evaluation split also appears in train/validation/dev. "
                "Do not silently fall back to an overlapping-speaker evaluation."
            )

    selected = select_records(
        eligible_rows, limit=limit, seed=seed, max_per_speaker=max_per_speaker
    )
    if not selected:
        raise RuntimeError("No evaluation rows selected after speaker-aware filtering.")

    # Import the production pipeline only after manifest validation and sampling.
    from pipeline import SpeechMindPipeline

    init_started = time.perf_counter()
    pipeline = SpeechMindPipeline()
    model_initialization_seconds = time.perf_counter() - init_started

    prediction_rows: list[dict] = []
    asr_metric_rows: list[dict] = []
    translation_metric_rows: list[dict] = []
    output_root = output_dir.resolve()
    run_started = time.perf_counter()

    for index, row in enumerate(selected, start=1):
        path = resolve_audio_path(row["audio_path"], audio_root)
        run_error = None
        result: dict = {}
        if not path.is_file():
            run_error = f"FileNotFoundError: {path}"
        else:
            try:
                result = pipeline.run(str(path))
            except Exception as error:
                run_error = f"{type(error).__name__}: {error}"

        hypothesis = str(result.get("transcript") or "").strip()
        asr_score = error_counts_multi_reference(row["transcript"], hypothesis)
        asr_metric_rows.append(asr_score)

        reference_meaning = str(row.get("reference_meaning_en") or "").strip()
        translation_score = None
        if reference_meaning:
            translated = (
                str(result.get("meaning_en") or "").strip()
                if result.get("meaning_en_available") else ""
            )
            translation_score = error_counts_multi_reference(
                reference_meaning, translated
            )
            translation_metric_rows.append(translation_score)

        timings = result.get("timings_sec") or {}
        emotion_label = canonical_emotion(row.get("emotion_label"))
        prediction_row = {
            "dataset": row["dataset"],
            "split": row["split"],
            "speaker_id": row["speaker_id"],
            "age_group": row.get("age_group", ""),
            "language": row.get("language", ""),
            "audio_path": row["audio_path"],
            "run_ok": not bool(run_error),
            "run_error": run_error or "",
            "reference_transcript": row["transcript"],
            "hypothesis_transcript": hypothesis,
            "asr_wer": asr_score["wer"],
            "asr_cer": asr_score["cer"],
            "detected_language": result.get("language", ""),
            "asr_model": result.get("asr_model", ""),
            "asr_context_terms": result.get("asr_context_terms", ""),
            "asr_context_source": result.get("asr_context_source", ""),
            "reference_meaning_en": reference_meaning,
            "meaning_en": result.get("meaning_en", ""),
            "meaning_en_available": result.get("meaning_en_available", False),
            "translation_wer": translation_score["wer"] if translation_score else None,
            "translation_cer": translation_score["cer"] if translation_score else None,
            "gold_emotion": emotion_label,
            "emotion_label": emotion_label,
            "asr_metrics": asr_score,
            "text_emotion": result.get("text_emotion", ""),
            "voice_emotion": result.get("voice_emotion", ""),
            "overall_emotion": result.get("overall_emotion", ""),
            "interpretation_status": result.get("interpretation_status", ""),
            "timings_sec": timings,
            "processing_time_sec": result.get("processing_time_sec"),
        }
        prediction_rows.append(prediction_row)

        print(
            f"[FullPipelineEval] {index}/{len(selected)} "
            f"[{row['dataset']}/{row.get('age_group') or row.get('language') or 'all'}] "
            f"WER={asr_score['wer']:.3f} CER={asr_score['cer']:.3f} "
            f"total={float(result.get('processing_time_sec') or 0.0):.1f}s"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions.csv"
    write_csv(predictions_path, prediction_rows)

    successful = [row for row in prediction_rows if row["run_ok"]]
    run_times = [
        float(row["processing_time_sec"])
        for row in successful if row["processing_time_sec"] is not None
    ]
    by_dataset = grouped_asr_metrics(prediction_rows, "dataset") if prediction_rows else {}
    group_column = (
        "age_group"
        if any(row.get("age_group") for row in prediction_rows)
        else "language"
    )
    by_group = grouped_asr_metrics(prediction_rows, group_column) if prediction_rows else {}
    summary = {
        "benchmark": "SpeechMind production full-pipeline evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": current_git_revision(),
        "manifest": str(manifest_path.resolve()),
        "audio_root": str(audio_root.resolve()),
        "split": split,
        "sampling": {
            "requested_limit": limit,
            "selected_samples": len(selected),
            "seed": seed,
            "max_per_speaker": max_per_speaker,
            "strict_speaker_holdout": strict_speaker_holdout,
            "speaker_split_report": split_report,
            "selection_note": (
                "At most the requested number of clips per speaker is selected. "
                "This is not a claim of speaker independence unless strict holdout "
                "is enabled and the manifest includes trustworthy train/validation "
                "speaker IDs."
            ),
        },
        "samples": {
            "selected": len(prediction_rows),
            "successful_pipeline_runs": len(successful),
            "failed_pipeline_runs": len(prediction_rows) - len(successful),
            "unique_speakers": len({
                (row["dataset"], row["speaker_id"]) for row in selected
            }),
        },
        "asr": {
            "micro": aggregate_error_counts(asr_metric_rows),
            "by_dataset": by_dataset,
            f"by_{group_column}": by_group,
        },
        "translation": (
            {
                "scored_samples": len(translation_metric_rows),
                "micro": aggregate_error_counts(translation_metric_rows),
                "note": "Scored only where reference_meaning_en is supplied in the manifest.",
            }
            if translation_metric_rows else {
                "scored_samples": 0,
                "micro": None,
                "status": "not_scored_no_reference_meaning_en_column_values",
            }
        ),
        "emotion": {
            "voice": classification_metrics(prediction_rows, "voice_emotion"),
            "text": classification_metrics(prediction_rows, "text_emotion"),
            "note": (
                "Scored only when gold emotion_label is supplied. Voice and text "
                "metrics are separate; uncertain/unavailable predictions are abstentions."
            ),
        },
        "performance": {
            "model_initialization_seconds": round(model_initialization_seconds, 3),
            "median_processing_seconds": (
                round(statistics.median(run_times), 3) if run_times else None
            ),
            "mean_processing_seconds": (
                round(statistics.mean(run_times), 3) if run_times else None
            ),
            "median_stage_timings": grouped_timing_medians(prediction_rows),
            "benchmark_elapsed_seconds": round(
                time.perf_counter() - run_started, 3
            ),
        },
        "predictions_csv": str(predictions_path),
        "quality_gates": {
            "asr_scored": bool(asr_metric_rows),
            "translation_scored": bool(translation_metric_rows),
            "emotion_scored": any(
                str(row.get("emotion_label") or "").strip() for row in selected
            ),
            "speaker_disjoint": split_report["speaker_disjoint_from_manifest_train_validation"]
            if split_report["training_validation_rows_in_manifest"] else None,
            "note": (
                "This report measures the supplied sample and references. Passing "
                "the runner is not equivalent to passing quality thresholds."
            ),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n[FullPipelineEval] Evaluation complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="CSV manifest with audio_path, transcript, speaker_id, split columns.",
    )
    parser.add_argument(
        "--audio-root", type=Path, default=PROJECT_ROOT / "data" / "hiacc",
        help="Base directory for relative audio_path values.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=40,
                        help="Maximum clips; use 0 for all eligible rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-speaker", type=int, default=1)
    parser.add_argument("--strict-speaker-holdout", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: data/benchmarks/full_pipeline/run-<UTC timestamp>.",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        PROJECT_ROOT / "data" / "benchmarks" / "full_pipeline" / f"run-{timestamp}"
    )
    try:
        run_evaluation(
            manifest_path=args.input,
            audio_root=args.audio_root,
            output_dir=output_dir,
            split=args.split,
            limit=args.limit,
            seed=args.seed,
            max_per_speaker=args.max_per_speaker,
            strict_speaker_holdout=args.strict_speaker_holdout,
        )
        return 0
    except Exception as error:
        print(f"\n[FullPipelineEval] ERROR: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
