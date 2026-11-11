"""Fine-tune MuRIL for SpeechMind's seven-way Hindi/Hinglish text emotion view.

The six non-neutral labels are trained as masked multi-label outputs. Neutral
is derived when no non-neutral label crosses its threshold. Hinglish HinEmo
rows carry lower sample weight and mask labels the source does not provide.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "google/muril-base-cased"
LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared emotion data is missing: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not str(row.get("text") or "").strip():
                continue
            if len(row.get("labels", [])) != len(LABELS):
                raise ValueError(f"Invalid six-label target in {path}:{line_number}")
            if len(row.get("label_mask", [])) != len(LABELS):
                raise ValueError(f"Invalid six-label mask in {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No usable records in {path}")
    return rows


def sample_hinemo(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("raw_label") or "unknown")].append(row)
    keys = sorted(groups)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    while len(selected) < limit:
        added = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
                added = True
        if not added:
            break
    return selected


class MaskedMultiLabelDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.encodings = tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = [row["labels"] for row in rows]
        self.masks = [row["label_mask"] for row in rows]
        self.weights = [float(row.get("source_weight", 1.0)) for row in rows]
        self.sources = [str(row.get("source", "unknown")) for row in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(values[index], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        item["label_mask"] = torch.tensor(self.masks[index], dtype=torch.float32)
        item["sample_weight"] = torch.tensor(self.weights[index], dtype=torch.float32)
        return item


def build_model(trainable_layers: int):
    id2label = {index: label for index, label in enumerate(LABELS)}
    label2id = {label: index for index, label in enumerate(LABELS)}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False

    base_model = model.base_model
    encoder = getattr(base_model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise RuntimeError(
            "Could not locate MuRIL encoder layers for partial fine-tuning."
        )
    trainable_layers = max(0, min(int(trainable_layers), len(layers)))
    for layer in (layers[-trainable_layers:] if trainable_layers else []):
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for name, parameter in model.named_parameters():
        if "classifier" in name or "pooler" in name:
            parameter.requires_grad = True
    return model


def compute_pos_weights(rows, device):
    positives = np.zeros(len(LABELS), dtype=np.float64)
    negatives = np.zeros(len(LABELS), dtype=np.float64)
    for row in rows:
        weight = float(row.get("source_weight", 1.0))
        for index, (label, mask) in enumerate(zip(row["labels"], row["label_mask"])):
            if int(mask):
                if int(label):
                    positives[index] += weight
                else:
                    negatives[index] += weight
    ratios = [
        min(8.0, max(1.0, negatives[i] / max(positives[i], 1.0)))
        for i in range(len(LABELS))
    ]
    return torch.tensor(ratios, dtype=torch.float32, device=device)


def calculate_source_metrics(rows, predictions) -> dict:
    sources = sorted({str(row.get("source", "unknown")) for row in rows})
    results = {}
    for source in sources:
        indices = [
            i for i, row in enumerate(rows)
            if str(row.get("source", "unknown")) == source
        ]
        y_true = np.asarray([rows[i]["labels"] for i in indices], dtype=np.int64)
        y_mask = np.asarray([rows[i]["label_mask"] for i in indices], dtype=np.int64)
        y_pred = np.asarray([predictions[i] for i in indices], dtype=np.int64)

        per_label = {}
        f1_values = []
        for label_index, label in enumerate(LABELS):
            keep = y_mask[:, label_index].astype(bool)
            if not keep.any():
                continue
            truth = y_true[keep, label_index]
            pred = y_pred[keep, label_index]
            score = float(f1_score(truth, pred, zero_division=0))
            f1_values.append(score)
            per_label[label] = {
                "support": int(keep.sum()),
                "precision": float(precision_score(truth, pred, zero_division=0)),
                "recall": float(recall_score(truth, pred, zero_division=0)),
                "f1": score,
            }

        exact_rows = y_mask.all(axis=1)
        neutral_metrics = None
        if exact_rows.any():
            truth_neutral = (y_true[exact_rows].sum(axis=1) == 0).astype(int)
            pred_neutral = (y_pred[exact_rows].sum(axis=1) == 0).astype(int)
            neutral_metrics = {
                "support": int(exact_rows.sum()),
                "precision": float(precision_score(
                    truth_neutral, pred_neutral, zero_division=0
                )),
                "recall": float(recall_score(
                    truth_neutral, pred_neutral, zero_division=0
                )),
                "f1": float(f1_score(
                    truth_neutral, pred_neutral, zero_division=0
                )),
            }

        results[source] = {
            "samples": len(indices),
            "known_label_macro_f1": (
                float(statistics.mean(f1_values)) if f1_values else None
            ),
            "per_label": per_label,
            "derived_neutral": neutral_metrics,
        }
    return results


@torch.no_grad()
def evaluate(model, loader, rows, device, thresholds):
    model.eval()
    all_probabilities = []
    for batch in loader:
        labels = batch.pop("labels")
        batch.pop("label_mask")
        batch.pop("sample_weight")
        inputs = {key: value.to(device) for key, value in batch.items()}
        logits = model(**inputs).logits
        all_probabilities.extend(torch.sigmoid(logits).cpu().tolist())

    predictions = []
    for probability_row in all_probabilities:
        predictions.append([
            int(float(probability_row[index]) >= thresholds[index])
            for index in range(len(LABELS))
        ])

    return {
        "sources": calculate_source_metrics(rows, predictions),
        "thresholds": {
            label: float(thresholds[index])
            for index, label in enumerate(LABELS)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/text_emotion/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/text_emotion_muril"))
    parser.add_argument("--max-hinemo-train", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--trainable-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    brighter_train = read_jsonl(args.data_dir / "brighter_hi_train.jsonl")
    brighter_dev = read_jsonl(args.data_dir / "brighter_hi_dev.jsonl")
    brighter_test = read_jsonl(args.data_dir / "brighter_hi_test.jsonl")
    hinemo_train_all = read_jsonl(args.data_dir / "hinemo_train.jsonl")
    hinemo_val = read_jsonl(args.data_dir / "hinemo_val.jsonl")
    hinemo_test = read_jsonl(args.data_dir / "hinemo_test.jsonl")
    hinemo_train = sample_hinemo(
        hinemo_train_all, args.max_hinemo_train, args.seed
    )
    train_rows = brighter_train + hinemo_train

    print(
        f"[TextEmotion] train rows={len(train_rows)} "
        f"(BRIGHTER={len(brighter_train)}, HinEmo={len(hinemo_train)}"
        f"/{len(hinemo_train_all)}); BRIGHTER dev={len(brighter_dev)}; "
        f"BRIGHTER test={len(brighter_test)}; device={device}"
    )
    print(
        "[TextEmotion] Multi-label MuRIL; only the six non-neutral labels are "
        "predicted. Neutral is derived when no label exceeds its threshold."
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = MaskedMultiLabelDataset(train_rows, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    dev_dataset = MaskedMultiLabelDataset(brighter_dev, tokenizer, args.max_length)
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = build_model(args.trainable_layers).to(device)
    pos_weights = compute_pos_weights(train_rows, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    history = []
    thresholds = [0.5] * len(LABELS)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            mask = batch.pop("label_mask").to(device)
            sample_weight = batch.pop("sample_weight").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            element_loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weights, reduction="none"
            )
            weighted_mask = mask * sample_weight.unsqueeze(1)
            denominator = weighted_mask.sum().clamp_min(1.0)
            loss = (element_loss * weighted_mask).sum() / denominator
            (loss / max(1, args.gradient_accumulation)).backward()
            losses.append(float(loss.detach().cpu()))

            update = (
                (step + 1) % max(1, args.gradient_accumulation) == 0
                or step + 1 == len(train_loader)
            )
            if update:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        dev_metrics = evaluate(model, dev_loader, brighter_dev, device, thresholds)
        brighter_dev_metrics = dev_metrics["sources"].get("BRIGHTER-Hindi", {})
        known_macro = brighter_dev_metrics.get("known_label_macro_f1") or 0.0
        neutral_f1 = (
            (brighter_dev_metrics.get("derived_neutral") or {}).get("f1") or 0.0
        )
        selection_score = (known_macro * 6.0 + neutral_f1) / 7.0
        row = {
            "epoch": epoch + 1,
            "mean_training_loss": statistics.mean(losses) if losses else None,
            "brighter_dev_known_label_macro_f1": known_macro,
            "brighter_dev_neutral_f1": neutral_f1,
            "selection_score": selection_score,
        }
        history.append(row)
        print("[TextEmotion] " + json.dumps(row))

        if selection_score > best_score:
            best_score = selection_score
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(args.output_dir)
    ).to(device)

    test_results = {}
    for name, rows in (
        ("brighter_hi_test", brighter_test),
        ("hinemo_gpt_test", hinemo_test),
        ("hinemo_gpt_validation", hinemo_val),
    ):
        dataset = MaskedMultiLabelDataset(rows, tokenizer, args.max_length)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        test_results[name] = evaluate(
            best_model, loader, rows, device, thresholds
        )

    metadata = {
        "model_type": "muril_text_emotion_multilabel",
        "task_type": "multilabel_six_plus_neutral",
        "backbone": MODEL_NAME,
        "labels": LABELS,
        "class_order": LABELS,
        "languages": ["hi", "hinglish"],
        "neutral_rule": "neutral if no non-neutral label meets its per-class threshold",
        "thresholds": {label: 0.5 for label in LABELS},
        "training_method": "partial_finetune_last_encoder_layers",
        "trainable_encoder_layers": args.trainable_layers,
        "epochs_requested": args.epochs,
        "best_validation_selection_score": best_score,
        "training_rows": {
            "brighter_hindi": len(brighter_train),
            "hinemo_gpt_sampled": len(hinemo_train),
            "hinemo_gpt_full_train": len(hinemo_train_all),
        },
        "dataset_manifest": str((args.data_dir / "dataset_manifest.json").resolve()),
        "limitations": [
            "HinEmo labels are GPT-generated and are down-weighted during training.",
            "HinEmo does not supervise fear, surprise, or neutral.",
            "BRIGHTER Hindi is multi-label and largely created emotion-bearing text, not spontaneous Hinglish dialogue.",
            "The model is experimental and not clinically or psychologically validated.",
        ],
    }
    (args.output_dir / "speechmind_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "text_emotion_metrics.json").write_text(
        json.dumps({
            "history": history,
            "test_and_validation": test_results,
            "selection_metric": "weighted average of BRIGHTER six-label macro-F1 and derived-neutral F1",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[TextEmotion] Training complete. Checkpoint: "
        + str(args.output_dir.resolve())
    )
    print(json.dumps(test_results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
