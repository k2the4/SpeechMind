"""Compare local Whisper and optional Bodhan ASR on a human-transcribed manifest.

Manifest CSV columns:
  audio_path,reference
Optional:
  language,split,speaker_id

Paths may be absolute or relative to the SpeechMind project directory.
WER and CER are computed over the complete reference/prediction corpus.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from models.asr import SpeechRecognizer
from utils.audio_utils import preprocess


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or "")).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, 1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, 1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            ))
        previous = current
    return previous[-1]


def score_pair(reference: str, hypothesis: str) -> dict:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    ref_words = ref.split()
    hyp_words = hyp.split()
    word_errors = edit_distance(ref_words, hyp_words)
    char_ref = list(ref.replace(" ", ""))
    char_hyp = list(hyp.replace(" ", ""))
    char_errors = edit_distance(char_ref, char_hyp)
    return {
        "word_errors": word_errors,
        "reference_words": len(ref_words),
        "character_errors": char_errors,
        "reference_characters": len(char_ref),
        "wer": word_errors / max(1, len(ref_words)),
        "cer": char_errors / max(1, len(char_ref)),
    }


def aggregate(rows: list[dict]) -> dict:
    word_errors = sum(row["word_errors"] for row in rows)
    reference_words = sum(row["reference_words"] for row in rows)
    char_errors = sum(row["character_errors"] for row in rows)
    reference_characters = sum(row["reference_characters"] for row in rows)
    return {
        "clips": len(rows),
        "wer": word_errors / max(1, reference_words),
        "cer": char_errors / max(1, reference_characters),
        "word_errors": word_errors,
        "reference_words": reference_words,
        "character_errors": char_errors,
        "reference_characters": reference_characters,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/asr_benchmark/manifest.csv")
    parser.add_argument("--backend", choices=("all", "whisper", "bodhan"), default="all")
    parser.add_argument("--output-dir", default="training_outputs/asr_benchmark")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"audio_path", "reference"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        records = [row for row in reader if row.get("audio_path") and row.get("reference")]

    if args.limit > 0:
        records = records[:args.limit]
    if not records:
        raise ValueError("The manifest contains no audio_path/reference rows.")

    project_root = Path.cwd()
    backends = ["whisper", "bodhan"] if args.backend == "all" else [args.backend]
    recognizers = {}
    for backend in backends:
        try:
            recognizers[backend] = SpeechRecognizer(backend=backend)
        except Exception as error:
            if backend == "bodhan":
                print(f"[Benchmark] Skipping Bodhan: {error}")
            else:
                raise

    if not recognizers:
        raise RuntimeError("No ASR backend is available for this benchmark.")

    predictions = []
    summary = {}
    for backend, recognizer in recognizers.items():
        rows = []
        print(f"\n[Benchmark] Running {backend} on {len(records)} recordings...")
        for index, item in enumerate(records, 1):
            audio_path = Path(item["audio_path"])
            if not audio_path.is_absolute():
                audio_path = project_root / audio_path
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Audio file in manifest row {index} does not exist: {audio_path}"
                )

            audio = preprocess(str(audio_path))
            result = recognizer.transcribe(audio)
            transcript = result.get("transcript", "")
            metrics = score_pair(item["reference"], transcript)
            row = {
                "backend": backend,
                "audio_path": str(audio_path),
                "language": item.get("language", ""),
                "split": item.get("split", ""),
                "speaker_id": item.get("speaker_id", ""),
                "reference": item["reference"],
                "prediction": transcript,
                **metrics,
            }
            rows.append(row)
            predictions.append(row)
            print(
                f"[{backend}] {index}/{len(records)} "
                f"WER={metrics['wer']:.3f} CER={metrics['cer']:.3f}"
            )
        summary[backend] = aggregate(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "predictions.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(predictions[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n[Benchmark] Aggregate results:")
    print(json.dumps(summary, indent=2))
    print(f"[Benchmark] Predictions: {output_csv}")
    print(f"[Benchmark] Metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
