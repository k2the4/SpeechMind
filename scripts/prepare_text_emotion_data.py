"""Prepare licensed Hindi/Hinglish emotion data for SpeechMind.

Downloads are explicit (this script is run by a BAT file, never at app startup).
Outputs preserve the original train/dev/test splits and per-label supervision
masks. Raw data is not committed to the repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BRIGHTER_DATASET = "brighter-dataset/BRIGHTER-emotion-categories"
BRIGHTER_REVISION = "419566a"
HINEMO_DATASET = "theanonymous524/hinemo"
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
BRIGHTER_LICENSE = "CC-BY-4.0 (as declared by Hugging Face dataset card)"
HINEMO_LICENSE = "CC-BY-4.0 (as declared by Hugging Face dataset card)"
HINEMO_ALIASES = {
    "anger": "anger", "angry": "anger",
    "disgust": "disgust", "disgusted": "disgust",
    "joy": "joy", "happy": "joy", "happiness": "joy",
    "sad": "sadness", "sadness": "sadness",
}


def _as_binary(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("", "0", "false", "no", "none"):
            return 0
        if value in ("1", "true", "yes"):
            return 1
    try:
        return int(float(value) > 0)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid binary emotion label: {value!r}")


def brighter_record(row: dict, split: str) -> dict | None:
    text = str(row.get("text") or "").strip()
    if not text:
        return None
    labels = [_as_binary(row.get(label, 0)) for label in EMOTIONS]
    return {
        "text": text,
        "labels": labels,
        "label_mask": [1] * len(EMOTIONS),
        "source_weight": 1.0,
        "source": "BRIGHTER-Hindi",
        "language": "hi",
        "split": split,
        "label_provenance": "human_annotated_with_dataset_quality_control",
    }


def hinemo_record(row: dict, split: str) -> dict | None:
    text = str(row.get("text") or "").strip()
    if not text:
        return None
    raw = str(row.get("gpt_emotion") or "").strip().lower()
    canonical = HINEMO_ALIASES.get(raw)
    if canonical is None:
        raise ValueError(f"Unmapped HinEmo gpt_emotion label: {raw!r}")
    labels = [int(label == canonical) for label in EMOTIONS]
    # This source only labels anger, disgust, joy and sadness. Fear and surprise
    # are unknown, not negative; neutral is also not annotated by this dataset.
    known = {"anger", "disgust", "joy", "sadness"}
    mask = [int(label in known) for label in EMOTIONS]
    return {
        "text": text,
        "labels": labels,
        "label_mask": mask,
        "source_weight": 0.35,
        "source": "HinEmo",
        "language": "hinglish",
        "split": split,
        "label_provenance": "gpt_generated_dataset_label",
        "raw_label": raw,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_and_prepare(output_dir: Path) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The Hugging Face datasets package is missing from the main .venv. "
            "Run UPDATE_SPEECHMIND_DEPS.bat first."
        ) from error

    print(f"[PrepareTextEmotion] Loading {BRIGHTER_DATASET} / hin ...")
    brighter = load_dataset(
        BRIGHTER_DATASET, "hin", revision=BRIGHTER_REVISION
    )
    print(f"[PrepareTextEmotion] Loading {HINEMO_DATASET} CSV splits ...")
    base = f"https://huggingface.co/datasets/{HINEMO_DATASET}/resolve/main"
    hinemo = load_dataset(
        "csv",
        data_files={
            "train": f"{base}/hinemo_train.csv",
            "validation": f"{base}/hinemo_val.csv",
            "test": f"{base}/hinemo_test.csv",
        },
    )

    manifest = {
        "taxonomy": EMOTIONS + ["neutral"],
        "formulation": "six independent emotion labels; neutral is no positive emotion",
        "brighter": {
            "dataset": BRIGHTER_DATASET,
            "revision": BRIGHTER_REVISION,
            "license": BRIGHTER_LICENSE,
            "splits": {},
        },
        "hinemo": {
            "dataset": HINEMO_DATASET,
            "license": HINEMO_LICENSE,
            "label_field": "gpt_emotion",
            "training_source_weight": 0.35,
            "supervised_emotions": ["anger", "disgust", "joy", "sadness"],
            "splits": {},
        },
    }

    for split in ("train", "dev", "test"):
        if split not in brighter:
            raise RuntimeError(
                f"BRIGHTER snapshot has no {split!r} split; found {list(brighter.keys())}"
            )
        rows = [
            record for item in brighter[split]
            if (record := brighter_record(item, split)) is not None
        ]
        if not rows:
            raise RuntimeError(f"BRIGHTER {split} split produced no usable rows.")
        write_jsonl(output_dir / f"brighter_hi_{split}.jsonl", rows)
        manifest["brighter"]["splits"][split] = len(rows)
        print(f"[PrepareTextEmotion] BRIGHTER {split}: {len(rows)}")

    for split in ("train", "validation", "test"):
        rows = [
            record for item in hinemo[split]
            if (record := hinemo_record(item, split)) is not None
        ]
        if not rows:
            raise RuntimeError(f"HinEmo {split} split produced no usable rows.")
        suffix = {"train": "train", "validation": "val", "test": "test"}[split]
        write_jsonl(output_dir / f"hinemo_{suffix}.jsonl", rows)
        manifest["hinemo"]["splits"][suffix] = len(rows)
        print(f"[PrepareTextEmotion] HinEmo {suffix}: {len(rows)}")

    manifest["total_prepared_records"] = sum(
        manifest[source]["splits"].get(split, 0)
        for source in ("brighter", "hinemo")
        for split in manifest[source]["splits"]
    )
    manifest["notes"] = [
        "BRIGHTER is multi-label; its no-emotion case is neutral.",
        "HinEmo labels are GPT-generated. The loss mask excludes fear and surprise.",
        "HinEmo does not annotate neutral; neutral is never inferred from its rows.",
        "Released validation and test splits are preserved; no random re-splitting occurs.",
        "The source datasets are not copied into the repository.",
    ]
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/text_emotion/prepared")
    )
    args = parser.parse_args()
    try:
        manifest = load_and_prepare(args.output_dir)
        print("\n[PrepareTextEmotion] Data preparation complete.")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"\n[PrepareTextEmotion] ERROR: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
