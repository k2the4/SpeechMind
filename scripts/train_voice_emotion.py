"""
Fine-tune WavLM for seven-class SpeechMind voice-emotion recognition.

RAVDESS clips are mapped to seven classes (calm and neutral are combined).
Train/validation/test partitions are grouped by actor to prevent speaker
leakage. Validation macro-F1 selects the checkpoint; the test set is used
exactly once for the final report.

Audio and weights stay on the local computer.
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
    str(Path(__file__).resolve().parents[1] / "data" / "ravdess"),
)
BASE_MODEL_NAME = "microsoft/wavlm-base-plus"


def parse_ravdess_filename(path):
    """Return a labeled row only for RAVDESS audio-only speech recordings."""
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
            "Expected names such as 03-01-05-01-02-01-12.wav."
        )
    return records


def assign_speaker_splits(records, seed=42):
    """Assign whole actors to splits; the same speaker never crosses splits."""
    speakers = sorted({row["speaker_id"] for row in records})
    if len(speakers) < 5:
        raise ValueError(
            f"Need at least 5 distinct RAVDESS actors for group splitting; found {len(speakers)}."
        )
    random.Random(seed).shuffle(speakers)
    test_count = max(1, round(len(speakers) * 0.20))
    validation_count = max(1, round(len(speakers) * 0.20))
    if test_count + validation_count >= len(speakers):
        raise ValueError("Not enough speakers to form train, validation, and test splits.")
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


def train_model(
    records,
    output_dir,
    model_dir,
    epochs,
    batch_size,
    backbone_learning_rate,
    head_learning_rate,
):
    import inspect
    import numpy as np
    import librosa
    import soundfile as sf
    import torch
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score
    )
    from transformers import (
        AutoModelForAudioClassification,
        AutoProcessor,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Training] Device: {device}")
    print(f"[Training] Backbone: {BASE_MODEL_NAME}")
    print(
        f"[Training] Learning rates - backbone: {backbone_learning_rate}; "
        f"emotion head: {head_learning_rate}"
    )
    if device == "cpu":
        print("[Training] CUDA unavailable; full WavLM fine-tuning may take several hours.")

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
                raise ValueError(f"Invalid or empty audio file: {row['path']}")
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
        raise ValueError("Train, validation, and test splits must all contain audio.")

    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(LABELS),
        label2id=LABEL_TO_ID,
        id2label={index: label for label, index in LABEL_TO_ID.items()},
        ignore_mismatched_sizes=True,
    )
    model.config.problem_type = "single_label_classification"
    # Keep the entire WavLM stack trainable. The previous run froze the
    # convolutional feature encoder and produced zero recall for two classes.

    train_dataset = AudioDataset(train_rows, processor)
    validation_dataset = AudioDataset(validation_rows, processor)
    test_dataset = AudioDataset(test_rows, processor)

    def compute_metrics(prediction):
        logits, targets = prediction
        predicted = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(targets, predicted)),
            "f1_macro": float(f1_score(
                targets,
                predicted,
                labels=list(range(len(LABELS))),
                average="macro",
                zero_division=0,
            )),
        }

    class LayerwiseTrainer(Trainer):
        """Use a smaller learning rate for WavLM and a larger one for its new head."""
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            groups = {
                (is_head, no_decay): []
                for is_head in (False, True)
                for no_decay in (False, True)
            }
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                lower_name = name.lower()
                is_head = "classifier" in lower_name or "projector" in lower_name
                no_decay = (
                    name.endswith(".bias")
                    or "layer_norm" in lower_name
                    or "layernorm" in lower_name
                )
                groups[(is_head, no_decay)].append(parameter)

            optimizer_groups = []
            for (is_head, no_decay), parameters in groups.items():
                if not parameters:
                    continue
                optimizer_groups.append({
                    "params": parameters,
                    "lr": head_learning_rate if is_head else backbone_learning_rate,
                    "weight_decay": 0.0 if no_decay else 0.01,
                })
            self.optimizer = torch.optim.AdamW(
                optimizer_groups, betas=(0.9, 0.999), eps=1e-8
            )
            return self.optimizer

    parameters = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = (
        "eval_strategy" if "eval_strategy" in parameters
        else "evaluation_strategy"
    )
    kwargs = {
        "output_dir": str(output_dir / "trainer_state"),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": 2,
        "warmup_ratio": 0.10,
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
        "data_seed": 42,
    }
    kwargs[strategy_key] = "epoch"
    training_args = TrainingArguments(**kwargs)
    trainer = LayerwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=AudioCollator(processor),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    # Validation is used for model selection and reported separately from test.
    validation_metrics = trainer.evaluate(validation_dataset)
    best_validation_macro_f1 = trainer.state.best_metric
    predictions = trainer.predict(test_dataset)
    predicted_ids = np.argmax(predictions.predictions, axis=-1)
    true_ids = predictions.label_ids
    metrics = {
        "accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "macro_f1": float(f1_score(
            true_ids, predicted_ids,
            labels=list(range(len(LABELS))),
            average="macro", zero_division=0,
        )),
        "classification_report": classification_report(
            true_ids, predicted_ids,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            true_ids, predicted_ids, labels=list(range(len(LABELS)))
        ).tolist(),
        "test_speakers": sorted({row["speaker_id"] for row in test_rows}),
        "test_files": len(test_rows),
        "labels": LABELS,
        "backbone": BASE_MODEL_NAME,
        "validation_metrics": validation_metrics,
        "best_validation_macro_f1": best_validation_macro_f1,
        "training_epochs_completed": trainer.state.epoch,
        "note": (
            "Held-out-speaker RAVDESS results. RAVDESS is acted English speech; "
            "these metrics do not establish accuracy on spontaneous or Indian-language speech."
        ),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    processor.save_pretrained(str(model_dir))
    metadata = {
        "backbone": BASE_MODEL_NAME,
        "labels": LABELS,
        "sample_rate": 16000,
        "best_validation_macro_f1": best_validation_macro_f1,
        "validation_metrics": validation_metrics,
        "trained_epochs": trainer.state.epoch,
        "validation_speakers": sorted({
            row["speaker_id"] for row in validation_rows
        }),
        "training_speakers": sorted({
            row["speaker_id"] for row in train_rows
        }),
    }
    (model_dir / "speechmind_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    report_path = output_dir / "test_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "best_validation_macro_f1": best_validation_macro_f1,
        "epochs_completed": trainer.state.epoch,
        "checkpoint": str(model_dir),
        "report": str(report_path),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default="training_outputs/wavlm_ravdess")
    parser.add_argument("--model-dir", default="checkpoints/voice_emotion_wavlm")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=5e-5)
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
            records,
            output_dir,
            args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            backbone_learning_rate=args.backbone_learning_rate,
            head_learning_rate=args.head_learning_rate,
        )


if __name__ == "__main__":
    main()
