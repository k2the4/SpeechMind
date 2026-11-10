"""Download a small, speaker-diverse IndicVoices Hindi validation sample.

The official dataset is streamed. This script saves only selected audio chunks
and a JSONL manifest; it never downloads the complete split.

Default: 100 clips, maximum one clip per speaker, shuffled buffer of 1,024.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from datasets import Audio, load_dataset


REPOSITORY = "ai4bharat/IndicVoices"
CONFIG = "hindi"
SPLIT = "valid"
DEFAULT_OUTPUT = Path("data/indicvoices/hindi/valid_sample")


def safe_audio_suffix(audio_path: str | None) -> str:
    suffix = Path(audio_path or "").suffix.lower()
    return suffix if suffix in {".flac", ".wav", ".mp3", ".ogg"} else ".flac"


def download_sample(
    output_dir: Path,
    max_records: int = 100,
    max_per_speaker: int = 1,
    seed: int = 42,
    buffer_size: int = 1024,
    overwrite: bool = False,
) -> dict:
    if max_records < 1 or max_per_speaker < 1 or buffer_size < 1:
        raise ValueError("Sample size, per-speaker cap, and buffer size must be positive.")

    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Choose another --output-dir "
                "or pass --overwrite to replace this sample."
            )
        shutil.rmtree(output_dir)

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    print(f"[IndicVoices] Streaming {REPOSITORY}/{CONFIG}/{SPLIT}...")
    print(
        f"[IndicVoices] Target={max_records}; max_per_speaker={max_per_speaker}; "
        f"shuffle_buffer={buffer_size}; seed={seed}"
    )
    print("[IndicVoices] No full-split download will be performed.")

    stream = load_dataset(
        REPOSITORY,
        CONFIG,
        split=SPLIT,
        streaming=True,
        token=True,
    )
    stream = stream.cast_column("audio_filepath", Audio(decode=False))
    stream = stream.shuffle(seed=seed, buffer_size=buffer_size)

    speaker_counts: dict[str, int] = {}
    total_bytes = 0
    total_duration = 0.0
    saved = 0
    skipped_missing_audio = 0

    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        for row in stream:
            speaker_id = str(row.get("speaker_id") or "").strip()
            if not speaker_id:
                continue
            if speaker_counts.get(speaker_id, 0) >= max_per_speaker:
                continue

            audio = row.get("audio_filepath") or {}
            audio_bytes = audio.get("bytes")
            if not audio_bytes:
                skipped_missing_audio += 1
                continue

            original_audio_path = str(audio.get("path") or "")
            suffix = safe_audio_suffix(original_audio_path)
            saved += 1
            filename = f"hi_{saved:04d}{suffix}"
            audio_path = audio_dir / filename
            audio_path.write_bytes(audio_bytes)

            duration = float(row.get("duration") or 0.0)
            record = {
                "sample_id": f"hi_{saved:04d}",
                "audio_file": str(Path("audio") / filename).replace("\\", "/"),
                "source_audio_path": original_audio_path,
                "dataset": REPOSITORY,
                "config": CONFIG,
                "split": SPLIT,
                "lang": row.get("lang", "hi"),
                "text": row.get("text"),
                "normalized": row.get("normalized") or row.get("text"),
                "verbatim": row.get("verbatim"),
                "unsanitized_normalized": row.get("unsanitized_normalized"),
                "unsanitized_verbatim": row.get("unsanitized_verbatim"),
                "duration": duration,
                "speaker_id": speaker_id,
                "scenario": row.get("scenario"),
                "task_name": row.get("task_name"),
                "gender": row.get("gender"),
                "age_group": row.get("age_group"),
                "state": row.get("state"),
                "district": row.get("district"),
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()

            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
            total_bytes += len(audio_bytes)
            total_duration += duration

            if saved % 10 == 0 or saved == max_records:
                print(
                    f"[IndicVoices] Saved {saved}/{max_records}; "
                    f"speakers={len(speaker_counts)}; "
                    f"audio={total_bytes / (1024 * 1024):.1f} MiB"
                )

            if saved >= max_records:
                break

    if saved == 0:
        raise RuntimeError(
            "No audio records were saved. Check gated-dataset access and the "
            "audio_filepath field returned by the streaming dataset."
        )

    summary = {
        "dataset": REPOSITORY,
        "config": CONFIG,
        "split": SPLIT,
        "sample_count": saved,
        "unique_speakers": len(speaker_counts),
        "max_per_speaker": max_per_speaker,
        "shuffle_seed": seed,
        "shuffle_buffer": buffer_size,
        "total_duration_seconds": round(total_duration, 3),
        "audio_megabytes": round(total_bytes / (1024 * 1024), 3),
        "skipped_missing_audio": skipped_missing_audio,
        "manifest": str(manifest_path),
        "license": "CC-BY-4.0",
        "note": "ASR validation only; this dataset split contains no emotion labels.",
    }
    (output_dir / "sample_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "ATTRIBUTION.txt").write_text(
        "Dataset: AI4Bharat IndicVoices\n"
        "Source: https://huggingface.co/datasets/ai4bharat/IndicVoices\n"
        "License: CC BY 4.0 (review and follow the dataset card terms).\n"
        "This local subset is for ASR validation. It is not emotion-labelled data.\n",
        encoding="utf-8",
    )
    print("\n[IndicVoices] Sample download complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--max-per-speaker", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        download_sample(
            output_dir=args.output_dir,
            max_records=args.max_records,
            max_per_speaker=args.max_per_speaker,
            seed=args.seed,
            buffer_size=args.buffer_size,
            overwrite=args.overwrite,
        )
        return 0
    except Exception as error:
        print(f"\n[IndicVoices] ERROR: {type(error).__name__}: {error}")
        print(
            "Check Hugging Face authentication in this project's .venv and "
            "confirm that IndicVoices access is granted."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
