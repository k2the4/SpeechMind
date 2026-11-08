"""
Fine-tune a speech-pretrained wav2vec2 backbone on RAVDESS speech audio.\nA fresh emotion head is trained to avoid reusing an emotion checkpoint already trained on RAVDESS.

This script recognizes the official RAVDESS filename convention and creates
speaker-independent train/validation/test splits. Audio and checkpoints stay
on the user's machine and are never uploaded to GitHub.

Examples:
  python scripts/train_voice_emotion.py --prepare-only
  python scripts/train_voice_emotion.py --data-dir "D:/datasets/Audio_Speech_Actors_01-24_16k"
  python scripts/train_voice_emotion.py --epochs 3 --batch-size 2
"""
import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
RAVDESS_TO_LABEL = {
    "01": "neutral", "02": "neutral", "03": "joy", "04": "sadness",
    "05": "anger", "06": "fear", "07": "disgust", "08": "surprise",
}
DEFAULT_DATA_DIR = os.environ.get(
    "SPEECHMIND_RAVDESS_DIR",
    r"C:\Users\kvasi\OneDrive\Desktop\Test Data\Audio_Speech_Actors_01-24_16k",
)
BASE_MODEL_NAME = "facebook/wav2vec2-base-960h"


def parse_ravdess_filename(path):
    """Return a record only for RAVDESS audio-only speech files."""
    parts = Path(path).stem.split("-")
    if len(parts) != 7 or not all(part.isdigit() for part in parts):
        return None
    modality, channel, emotion, intensity, statement, repetition, actor = parts
    if modality != "03" or channel != "01" or emotion not in RAVDESS_TO_LABEL:
        return None
    if not 1 <= int(actor) <= 24:
        return None
    return {
        "path": str(Path(path).resolve()),
        "label": RAVDESS_TO_LABEL[emotion],
        "speaker_id": f"actor_{int(actor):02d}",
        "source_emotion_code": emotion,
    }


def build_manifest(data_dir):
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"RAVDESS folder not found: {root}. Set --data-dir to the folder containing actor folders."
        )
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".wav":
            row = parse_ravdess_filename(path)
            if row:
                records.append(row)
    if not records:
        raise ValueError(
            f"No RAVDESS speech WAV files found under {root}. "
            "Expected filenames such as 03-01-05-01-02-01-12.wav."
        )
    return records


def assign_speaker_splits(records, seed=42):
    """Split by actor, so a person's voice cannot appear in multiple sets."""
    speakers = sorted({row["speaker_id"] for row in records})
    if len(speakers) < 5:
        raise ValueError(
            f"Need at least 5 distinct RAVDESS actors for group splits; found {len(speakers)}."
        )
    random.Random(seed).shuffle(speakers)
    test_count = max(1, round(len(speakers) * 0.20))
    validation_count = max(1, round(len(speakers) * 0.20))
    if test_count + validation_count >= len(speakers):
        raise ValueError("Not enough speakers for train, validation, and test splits.")
    test_speakers = set(speakers[:test_count])
    validation_speakers = set(speakers[test_count:test_count + validation_count])
    output = []
    for row in records:
        split = (
            "test" if row["speaker_id"] in test_speakers
            else "validation" if row["speaker_id"] in validation_speakers
            else "train"
        )
        output.append({**row, "split": split})
    return output


def summarize(records):
    result = {}
    for split in ("train", "validation", "test"):
        subset = [row for row in records if row.get("split") == split]
        result[split] = {
            "files": len(subset),
            "speakers": sorted({row["speaker_id"] for row in subset}),
            "labels": dict(sorted(Counter(row["label"] for row in subset).items())),
        }
    return result


def write_manifest(records, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "label", "speaker_id", "source_emotion_code", "split"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return destination


def train_model(records, output_dir, model_dir, epochs, batch_size, learning_rate):
    import inspect
    import numpy as np
    import librosa
    import soundfile as sf
    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from transformers import AutoModelForAudioClassification, AutoProcessor, Trainer, TrainingArguments

    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Training] Device: {device}")
    if device == "cpu":
        print("[Training] CUDA unavailable; CPU fine-tuning may be slow.")

    class AudioDataset(torch.utils.data.Dataset):
        def __init__(self, rows, processor):
            self.rows = rows
            self.processor = processor

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            waveform, sample_rate = sf.read(
                row["path"], dtype="float32", always_2d=False
            )
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            if waveform.size == 0 or not np.isfinite(waveform).all():
                raise ValueError(f"Invalid or empty audio: {row['path']}")
            if sample_rate != 16000:
                waveform = librosa.resample(
                    waveform, orig_sr=sample_rate, target_sr=16000
                )
            encoded = self.processor(
                waveform.astype(np.float32, copy=False),
                sampling_rate=16000,
                return_tensors="pt",
            )
            return {
                "input_values": encoded["input_values"][0],
                "labels": LABEL_TO_ID[row["label"]],
            }

    class AudioCollator:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, features):
            batch = self.processor.pad(
                {"input_values": [item["input_values"] for item in features]},
                padding=True,
                return_tensors="pt",
            )
            batch["labels"] = torch.tensor(
                [item["labels"] for item in features], dtype=torch.long
            )
            return batch

    train_rows = [row for row in records if row["split"] == "train"]
    validation_rows = [row for row in records if row["split"] == "validation"]
    test_rows = [row for row in records if row["split"] == "test"]
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("Train, validation, and test must all contain audio.")

    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(LABELS),
        label2id=LABEL_TO_ID,
        id2label={index: label for label, index in LABEL_TO_ID.items()},
        ignore_mismatched_sizes=True,
    )
    model.config.problem_type = "single_label_classification"
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    train_dataset = AudioDataset(train_rows, processor)
    validation_dataset = AudioDataset(validation_rows, processor)
    test_dataset = AudioDataset(test_rows, processor)

    def compute_metrics(prediction):
        logits, targets = prediction
        predicted = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(targets, predicted)),
            "f1_macro": float(f1_score(
                targets, predicted, labels=list(range(len(LABELS))),
                average="macro", zero_division=0
            )),
        }

    parameters = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = (
        "eval_strategy" if "eval_strategy" in parameters
        else "evaluation_strategy"
    )
    kwargs = {
        "output_dir": str(output_dir / "trainer_state"),
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": 2,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "save_total_limit": 2,
        "dataloader_num_workers": 0,
        "fp16": device == "cuda",
        "report_to": "none",
        "seed": 42,
    }
    kwargs[strategy_key] = "epoch"
    training_args = TrainingArguments(**kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=AudioCollator(processor),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    predictions = trainer.predict(test_dataset)
    predicted_ids = np.argmax(predictions.predictions, axis=-1)
    true_ids = predictions.label_ids
    metrics = {
        "accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "macro_f1": float(f1_score(
            true_ids, predicted_ids, labels=list(range(len(LABELS))),
            average="macro", zero_division=0
        )),
        "classification_report": classification_report(
            true_ids, predicted_ids, labels=list(range(len(LABELS))),
            target_names=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            true_ids, predicted_ids, labels=list(range(len(LABELS)))
        ).tolist(),
        "test_speakers": sorted({row["speaker_id"] for row in test_rows}),
        "test_files": len(test_rows),
        "labels": LABELS,
        "note": (
            "Held-out-speaker RAVDESS results. RAVDESS is acted English speech; "
            "these metrics do not establish accuracy on spontaneous or Indian-language speech.",
            "training_backbone": BASE_MODEL_NAME,
            "emotion_head": "newly initialized before RAVDESS fine-tuning",
        ),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    processor.save_pretrained(str(model_dir))
    report_path = output_dir / "test_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "checkpoint": str(model_dir),
        "report": str(report_path),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default="training_outputs/ravdess_voice")
    parser.add_argument("--model-dir", default="checkpoints/voice_emotion_wav2vec2")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    records = assign_speaker_splits(build_manifest(args.data_dir))
    output_dir = Path(args.output_dir)
    manifest_path = write_manifest(records, output_dir / "manifest_with_splits.csv")
    summary = summarize(records)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[Dataset] Found {len(records)} labeled RAVDESS speech clips.")
    print(json.dumps(summary, indent=2))
    print(f"[Dataset] Manifest saved: {manifest_path}")
    if not args.prepare_only:
        train_model(
            records, output_dir, args.model_dir,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()
