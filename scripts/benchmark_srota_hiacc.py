"""Run a small local CPU benchmark of Srota on the official HiACC test split.

Uses an isolated .venv_srota environment because qwen-asr requires a different
Transformers version from SpeechMind's existing IndicConformer/voice venv.
The first run downloads ~0.8B model weights. Default benchmark is 20 test
utterances, balanced across adult/children where possible.
"""
from __future__ import annotations

import argparse
import csv
import json
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

from scripts.asr_metrics import aggregate_error_counts, error_counts

DATA_ROOT = PROJECT_ROOT / "data" / "hiacc"
MANIFEST = DATA_ROOT / "manifest.csv"
MODEL_ID = "moorlee/qwen3-asr-0.6b-hinglish"
MAX_CLIP_SECONDS = 30.0


def resolve_audio_path(row: dict) -> Path:
    """Manifest audio paths are relative to data/hiacc, not the project root."""
    return DATA_ROOT / row["audio_path"]


def read_test_records() -> list[dict]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"{MANIFEST} not found. Run PREPARE_HIACC_DATASET.bat first."
        )
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("split", "").lower() == "test"]
    if not rows:
        raise RuntimeError("HiACC manifest has no rows in the test split.")
    return rows


def select_records(
    records: list[dict], limit: int, seed: int, max_per_speaker: int
) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[row.get("age_group") or "unknown"].append(row)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[dict] = []
    speaker_counts: dict[str, int] = {}
    group_names = sorted(groups, key=lambda key: (key != "adult", key != "children", key))
    while len(selected) < limit and group_names:
        any_added = False
        for group_name in list(group_names):
            pool = groups[group_name]
            while pool:
                row = pool.pop()
                speaker = row.get("speaker_id", "").strip()
                if speaker and speaker_counts.get(speaker, 0) >= max_per_speaker:
                    continue
                if speaker:
                    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                selected.append(row)
                any_added = True
                break
            if not pool:
                group_names.remove(group_name)
            if len(selected) >= limit:
                break
        if not any_added:
            break
    return selected


def script_label(text: str) -> str:
    scripts = set()
    for char in str(text or ""):
        name = unicodedata.name(char, "")
        if "DEVANAGARI" in name:
            scripts.add("Devanagari")
        elif "LATIN" in name:
            scripts.add("Latin")
        elif "ARABIC" in name:
            scripts.add("Arabic")
    return "+".join(sorted(scripts)) if scripts else "empty_or_other"


def extract_text(result) -> str:
    if isinstance(result, str):
        return result.strip()
    if hasattr(result, "text"):
        return str(result.text or "").strip()
    if isinstance(result, dict):
        return str(result.get("text") or result.get("transcription") or "").strip()
    return str(result).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-speaker", type=int, default=1)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "benchmarks" / "srota")
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        import torch
        from qwen_asr import Qwen3ASRModel

        torch.set_num_threads(max(1, args.threads))
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))
        records = select_records(
            read_test_records(), max(1, args.limit), args.seed,
            max(1, args.max_per_speaker),
        )
        if not records:
            raise RuntimeError("No HiACC test records selected.")

        print(f"[Srota] Model: {MODEL_ID}")
        print(f"[Srota] Test utterances selected: {len(records)}")
        print("[Srota] Device: CPU / float32 (GT 730 is not used)")
        print("[Srota] Loading model. First run downloads the model weights...")
        init_start = time.perf_counter()
        model = Qwen3ASRModel.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
            device_map="cpu",
            max_inference_batch_size=1,
            max_new_tokens=512,
        )
        initialization_seconds = time.perf_counter() - init_start
        print(f"[Srota] Model initialization: {initialization_seconds:.1f}s")

        output_rows = []
        metric_rows = []
        by_group = defaultdict(list)
        scripts = defaultdict(int)
        inference_times = []
        inference_start = time.perf_counter()

        for index, row in enumerate(records, start=1):
            audio_path = resolve_audio_path(row)
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing HiACC audio file: {audio_path}")
            import soundfile as sf
            info = sf.info(str(audio_path))
            duration = float(info.duration)
            if duration > MAX_CLIP_SECONDS:
                raise ValueError(
                    f"HiACC clip {audio_path.name} is {duration:.1f}s; "
                    f"Srota adapter limit is {MAX_CLIP_SECONDS:.0f}s. "
                    "Add explicit natural-boundary chunking before benchmarking."
                )

            clip_start = time.perf_counter()
            result = model.transcribe(audio=str(audio_path), language=None)
            hypothesis = extract_text(result[0] if isinstance(result, (list, tuple)) else result)
            inference_seconds = time.perf_counter() - clip_start
            inference_times.append(inference_seconds)

            reference = str(row.get("transcript") or "")
            metrics = error_counts(reference, hypothesis)
            metric_rows.append(metrics)
            age_group = row.get("age_group") or "unknown"
            by_group[age_group].append(metrics)
            script = script_label(hypothesis)
            scripts[script] += 1

            output_rows.append({
                "audio_path": row["audio_path"],
                "audio_filename": row.get("audio_filename"),
                "age_group": age_group,
                "speaker_id": row.get("speaker_id"),
                "reference": reference,
                "hypothesis": hypothesis,
                "hypothesis_script": script,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "duration_seconds": duration,
                "inference_seconds": round(inference_seconds, 3),
            })
            print(
                f"[Srota] {index}/{len(records)} | {age_group} | "
                f"{duration:.1f}s audio | inference {inference_seconds:.1f}s"
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = args.output_dir / "predictions.csv"
        with predictions_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)

        inference_elapsed = time.perf_counter() - inference_start
        summary = {
            "dataset": "HiACC",
            "dataset_record": "https://zenodo.org/records/15551669",
            "dataset_license": "CC-BY-4.0 as listed on Zenodo",
            "split": "test",
            "model": MODEL_ID,
            "model_license": "Apache-2.0",
            "device": "cpu",
            "dtype": "float32",
            "samples": len(output_rows),
            "age_groups": {
                group: aggregate_error_counts(group_metrics)
                for group, group_metrics in by_group.items()
            },
            "overall": aggregate_error_counts(metric_rows),
            "hypothesis_script_counts": dict(scripts),
            "model_initialization_seconds": round(initialization_seconds, 3),
            "clip_inference_elapsed_seconds": round(inference_elapsed, 3),
            "mean_inference_seconds": round(statistics.mean(inference_times), 3),
            "median_inference_seconds": round(statistics.median(inference_times), 3),
            "total_elapsed_seconds": round(time.perf_counter() - started, 3),
            "predictions_csv": str(predictions_path),
            "notes": [
                "This is a small local CPU smoke benchmark, not a full reproduction of the Srota paper result.",
                "The HiACC dataset test split is used; model card notes speaker overlap across HiACC train/validation/test.",
                "WER/CER are calculated with SpeechMind's NFKC/lowercase/punctuation normalization.",
                "The benchmark tests code-switched Hindi-English speech only; it does not validate other Indian languages.",
                "No hosted API is used. Model weights download from the public Hugging Face repository.",
            ],
        }
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n[Srota] Benchmark complete.")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    except Exception as error:
        print(f"\n[Srota] ERROR: {type(error).__name__}: {error}")
        print(
            "If the model could not load, keep the existing .venv untouched. "
            "The Srota dependencies are isolated in .venv_srota."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
