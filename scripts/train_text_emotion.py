"""
Fine-tune MuRIL for SpeechMind's seven-way text-emotion taxonomy.

Input CSV: one primary human-annotated emotion per text row.
Required columns: text, emotion. Optional column: language (ISO code such as en, hi).
On CPU, only the final two MuRIL encoder layers and classification head are trained.
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "google/muril-base-cased"
# HinEmo-9 does not annotate disgust; its Love, Nostalgia, and Devotion
# categories are distinct and are not collapsed into the seven-way voice taxonomy.
LABELS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ALIASES = {
    "joy": "joy", "happy": "joy", "happiness": "joy", "excited": "joy",
    "excitement": "joy", "love": "joy", "gratitude": "joy", "optimism": "joy",
    "amusement": "joy", "admiration": "joy", "celebration": "joy",
    "sad": "sadness", "sadness": "sadness", "grief": "sadness",
    "disappointment": "sadness", "disappointed": "sadness", "remorse": "sadness",
    "anger": "anger", "angry": "anger", "annoyance": "anger",
    "annoyed": "anger", "frustration": "anger", "frustrated": "anger",
    "rage": "anger",
    "fear": "fear", "fearful": "fear", "nervous": "fear",
    "nervousness": "fear", "worried": "fear", "worry": "fear",
    "concern": "fear", "anxiety": "fear",
    "surprise": "surprise", "surprised": "surprise",
    "disgust": "disgust", "disgusted": "disgust",
    "neutral": "neutral", "calm": "neutral",
}


class EncodedTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(values[index], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def load_csv(path, text_column, label_column, language_column):
    texts, labels, languages = [], [], set()
    unknown = Counter()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {text_column, label_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            text = str(row.get(text_column, "")).strip()
            raw_label = str(row.get(label_column, "")).strip().lower()
            canonical = ALIASES.get(raw_label)
            if not text or not canonical:
                if raw_label:
                    unknown[raw_label] += 1
                continue
            texts.append(text)
            labels.append(LABEL_TO_ID[canonical])
            if language_column and row.get(language_column):
                languages.add(str(row[language_column]).strip().lower())

    if unknown:
        raise ValueError(
            "Unmapped emotion labels were found: "
            + ", ".join(f"{key} ({count})" for key, count in sorted(unknown.items()))
            + ". Convert labels to the seven SpeechMind classes before training."
        )
    if not texts:
        raise ValueError("No usable rows found in the CSV.")
    return texts, labels, sorted(languages or {"en"})


def build_model():
    id2label = {index: label for index, label in enumerate(LABELS)}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False

    base_model = model.base_model
    encoder = getattr(base_model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise RuntimeError(
            "Could not locate MuRIL encoder layers to apply the CPU-friendly "
            "partial fine-tuning strategy."
        )
    for layer in layers[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for name, parameter in model.named_parameters():
        if "classifier" in name or "pooler" in name:
            parameter.requires_grad = True
    return model


def evaluate(model, loader, device):
    model.eval()
    truth, predicted = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predicted.extend(logits.argmax(dim=-1).cpu().tolist())
            truth.extend(labels.cpu().tolist())
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(
            truth, predicted, labels=list(range(len(LABELS))),
            average="macro", zero_division=0
        )),
        "classification_report": classification_report(
            truth, predicted, labels=list(range(len(LABELS))),
            target_names=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=list(range(len(LABELS)))
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/text_emotion/text_emotion.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="emotion")
    parser.add_argument("--language-column", default="language")
    parser.add_argument("--output-dir", default="checkpoints/text_emotion_muril")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts, labels, languages = load_csv(
        args.csv, args.text_column, args.label_column, args.language_column
    )
    counts = Counter(labels)
    missing_or_small = [
        LABELS[index] for index in range(len(LABELS)) if counts[index] < 20
    ]
    if missing_or_small:
        raise ValueError(
            "At least 20 labeled examples per emotion are required for a "
            "meaningful stratified split. Insufficient classes: "
            + ", ".join(missing_or_small)
        )

    x_train, x_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.20, random_state=args.seed, stratify=labels
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=args.seed, stratify=y_temp
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = EncodedTextDataset(
        x_train, y_train, tokenizer, args.max_length
    )
    validation_dataset = EncodedTextDataset(
        x_validation, y_validation, tokenizer, args.max_length
    )
    test_dataset = EncodedTextDataset(
        x_test, y_test, tokenizer, args.max_length
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = build_model().to(device)
    train_counts = Counter(y_train)
    class_weights = torch.tensor(
        [
            len(y_train) / (len(LABELS) * train_counts[index])
            for index in range(len(LABELS))
        ],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    history = []

    print(
        f"[TextEmotion] {len(texts)} rows; train={len(x_train)}, "
        f"validation={len(x_validation)}, test={len(x_test)}; "
        f"languages={languages}; device={device}"
    )
    print(f"[TextEmotion] Training only final 2 MuRIL encoder layers + classifier.")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            labels_batch = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            loss = F.cross_entropy(logits, labels_batch, weight=class_weights)
            (loss / args.gradient_accumulation).backward()
            running_loss += float(loss.detach().cpu())

            is_update_step = (
                (step + 1) % args.gradient_accumulation == 0
                or step + 1 == len(train_loader)
            )
            if is_update_step:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        validation = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch + 1,
            "mean_training_loss": running_loss / max(1, len(train_loader)),
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
        }
        history.append(row)
        print("[TextEmotion] " + json.dumps(row))

        if validation["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation["macro_f1"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(output_dir)
    ).to(device)
    test_metrics = evaluate(best_model, test_loader, device)
    metadata = {
        "model_type": "muril_text_emotion_sequence_classifier",
        "backbone": MODEL_NAME,
        "labels": LABELS,
        "class_order": LABELS,
        "languages": languages,
        "max_length": args.max_length,
        "training_method": "partial_finetune_last_two_encoder_layers",
        "epochs_requested": args.epochs,
        "best_validation_macro_f1": best_macro_f1,
        "train_rows": len(x_train),
        "validation_rows": len(x_validation),
        "test_rows": len(x_test),
    }
    (output_dir / "speechmind_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "text_emotion_metrics.json").write_text(
        json.dumps({"history": history, "test": test_metrics}, indent=2),
        encoding="utf-8",
    )
    print(
        "[TextEmotion] Test results: "
        + json.dumps({
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "checkpoint": str(output_dir),
        })
    )


if __name__ == "__main__":
    main()
