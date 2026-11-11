"""Train masked multi-label MuRIL on prepared BRIGHTER-Hindi and HinEmo JSONL.

Uses the datasets' released train/dev/test splits. HinEmo's unknown labels are
excluded from loss using label_mask; source_weight down-weights its GPT labels.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "google/muril-base-cased"
LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def read_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Prepared split not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not str(row.get("text") or "").strip():
                    continue
                labels = row.get("labels")
                mask = row.get("label_mask")
                if not isinstance(labels, list) or len(labels) != len(LABELS):
                    raise ValueError(f"{path}:{line_number}: expected six labels")
                if not isinstance(mask, list) or len(mask) != len(LABELS):
                    raise ValueError(f"{path}:{line_number}: expected six label_mask values")
                row["labels"] = [float(value) for value in labels]
                row["label_mask"] = [float(value) for value in mask]
                row["source_weight"] = float(row.get("source_weight", 1.0))
                rows.append(row)
    if not rows:
        raise ValueError("No usable prepared records were found.")
    return rows


class EmotionDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.encodings = tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        item = {
            key: torch.tensor(values[index], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(row["labels"], dtype=torch.float32)
        item["label_mask"] = torch.tensor(row["label_mask"], dtype=torch.float32)
        item["source_weight"] = torch.tensor(row["source_weight"], dtype=torch.float32)
        return item


def masked_weighted_bce(logits, labels, label_mask, source_weight):
    element_loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    weights = label_mask * source_weight.unsqueeze(1)
    denominator = weights.sum().clamp_min(1.0)
    return (element_loss * weights).sum() / denominator


def build_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    encoder = getattr(model.base_model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise RuntimeError("Could not locate MuRIL encoder layers.")
    for layer in layers[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for name, parameter in model.named_parameters():
        if "classifier" in name or "pooler" in name:
            parameter.requires_grad = True
    return model


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    truths, predictions, known_masks = [], [], []
    total_loss = 0.0
    batches = 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        masks = batch.pop("label_mask").to(device)
        source_weights = batch.pop("source_weight").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        logits = model(**inputs).logits
        total_loss += float(masked_weighted_bce(
            logits, labels, masks, source_weights
        ).cpu())
        batches += 1
        predictions.extend((torch.sigmoid(logits) >= threshold).cpu().numpy())
        truths.extend(labels.cpu().numpy())
        known_masks.extend(masks.cpu().numpy())

    truth = np.asarray(truths, dtype=np.int32)
    predicted = np.asarray(predictions, dtype=np.int32)
    known = np.asarray(known_masks, dtype=np.int32)
    # Report only categories with at least one positive and one negative
    # known label in the held-out split.
    per_label = {}
    f1_values = []
    for index, label in enumerate(LABELS):
        known_rows = known[:, index] > 0
        y_true = truth[known_rows, index]
        y_pred = predicted[known_rows, index]
        score = float(f1_score(y_true, y_pred, zero_division=0))
        per_label[label] = score
        if len(np.unique(y_true)) > 1:
            f1_values.append(score)
    return {
        "loss": total_loss / max(1, batches),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_label_f1": per_label,
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/text_emotion/prepared")
    parser.add_argument("--output-dir", default="checkpoints/text_emotion_muril")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    train_rows = read_jsonl([
        data_dir / "brighter_hi_train.jsonl",
        data_dir / "hinemo_train.jsonl",
    ])
    validation_rows = read_jsonl([
        data_dir / "brighter_hi_dev.jsonl",
        data_dir / "hinemo_val.jsonl",
    ])
    test_rows = read_jsonl([
        data_dir / "brighter_hi_test.jsonl",
        data_dir / "hinemo_test.jsonl",
    ])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(
        EmotionDataset(train_rows, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    validation_loader = DataLoader(
        EmotionDataset(validation_rows, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        EmotionDataset(test_rows, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = build_model().to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    print(
        f"[TextEmotion] train={len(train_rows)} validation={len(validation_rows)} "
        f"test={len(test_rows)} device={device} labels={LABELS}",
        flush=True,
    )
    print(
        "[TextEmotion] MuRIL partial fine-tuning: final 2 encoder layers + head; "
        "masked weighted BCE; source splits preserved.",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            masks = batch.pop("label_mask").to(device)
            source_weights = batch.pop("source_weight").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = masked_weighted_bce(logits, labels, masks, source_weights)
            (loss / args.gradient_accumulation).backward()
            running_loss += float(loss.detach().cpu())
            if (
                (step + 1) % args.gradient_accumulation == 0
                or step + 1 == len(train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        validation = evaluate(model, validation_loader, device)
        print(json.dumps({
            "epoch": epoch + 1,
            "training_loss": running_loss / max(1, len(train_loader)),
            "validation": validation,
        }), flush=True)
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(output_dir)
    ).to(device)
    test_metrics = evaluate(best_model, test_loader, device)
    metadata = {
        "model_type": "muril_text_emotion_sequence_classifier",
        "task_type": "multilabel_six_plus_neutral",
        "backbone": MODEL_NAME,
        "labels": LABELS,
        "class_order": LABELS,
        "languages": ["hi", "hinglish"],
        "max_length": args.max_length,
        "training_method": "partial_finetune_last_two_encoder_layers",
        "training_sources": ["BRIGHTER-Hindi", "HinEmo"],
        "hinemo_labels_are_gpt_generated": True,
        "neutral_policy": "neutral is selected only when no non-neutral threshold is met",
        "thresholds": {label: 0.5 for label in LABELS},
        "best_validation_macro_f1": best_f1,
        "test_metrics": test_metrics,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "seed": args.seed,
    }
    (output_dir / "speechmind_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "text_emotion_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[TextEmotion] Saved checkpoint and held-out metrics:", flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
