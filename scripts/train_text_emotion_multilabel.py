"""CPU-practical first-pass MuRIL multi-label emotion training.

Uses all BRIGHTER-Hindi train rows plus a deterministic HinEmo subset.
The MuRIL encoder is frozen; only the classification head is trained.
This creates a baseline checkpoint quickly enough to validate the full pipeline.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "google/muril-base-cased"
LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared split not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("text") or "").strip():
                continue
            labels, mask = row.get("labels"), row.get("label_mask")
            if not isinstance(labels, list) or len(labels) != len(LABELS):
                raise ValueError(f"{path}:{line_number}: expected six labels")
            if not isinstance(mask, list) or len(mask) != len(LABELS):
                raise ValueError(f"{path}:{line_number}: expected six label_mask values")
            row["labels"] = [float(x) for x in labels]
            row["label_mask"] = [float(x) for x in mask]
            row["source_weight"] = float(row.get("source_weight", 1.0))
            rows.append(row)
    if not rows:
        raise ValueError(f"No usable records in {path}")
    return rows


def deterministic_subset(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[index] for index in indices]


class EmotionDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
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


def build_positive_weights(rows):
    labels = torch.tensor([row["labels"] for row in rows], dtype=torch.float32)
    masks = torch.tensor([row["label_mask"] for row in rows], dtype=torch.float32)
    source = torch.tensor([row["source_weight"] for row in rows], dtype=torch.float32).unsqueeze(1)
    weights = masks * source
    positives = (labels * weights).sum(dim=0)
    known = weights.sum(dim=0)
    negatives = (known - positives).clamp_min(0.0)
    # Prevent the all-negative solution while bounding rare-label amplification.
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(min=1.0, max=10.0)
    stats = {
        label: {
            "weighted_positives": round(float(positives[i]), 2),
            "weighted_negatives": round(float(negatives[i]), 2),
            "positive_weight": round(float(pos_weight[i]), 3),
        }
        for i, label in enumerate(LABELS)
    }
    return pos_weight, stats


def masked_weighted_bce(logits, labels, label_mask, source_weight, pos_weight):
    element_loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none", pos_weight=pos_weight
    )
    weights = label_mask * source_weight.unsqueeze(1)
    return (element_loss * weights).sum() / weights.sum().clamp_min(1.0)


def build_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        raise RuntimeError("MuRIL classifier head was not found.")
    for parameter in classifier.parameters():
        parameter.requires_grad = True
    return model


@torch.no_grad()
def collect_predictions(model, loader, device, pos_weight):
    model.eval()
    truth_rows, probability_rows, mask_rows = [], [], []
    total_loss, batches = 0.0, 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        masks = batch.pop("label_mask").to(device)
        weights = batch.pop("source_weight").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        logits = model(**inputs).logits
        total_loss += float(
            masked_weighted_bce(logits, labels, masks, weights, pos_weight).cpu()
        )
        batches += 1
        truth_rows.extend(labels.cpu().numpy())
        probability_rows.extend(torch.sigmoid(logits).cpu().numpy())
        mask_rows.extend(masks.cpu().numpy())
    return (
        np.asarray(truth_rows, dtype=np.int32),
        np.asarray(probability_rows, dtype=np.float32),
        np.asarray(mask_rows, dtype=np.int32),
        total_loss / max(1, batches),
    )


def calibrate_thresholds(truth, probabilities, known):
    """Choose per-label F1 thresholds on validation only, never on the test set."""
    thresholds = {}
    for index, label in enumerate(LABELS):
        selected = known[:, index] > 0
        y_true = truth[selected, index]
        y_prob = probabilities[selected, index]
        if not len(y_true) or len(np.unique(y_true)) < 2:
            thresholds[label] = 0.5
            continue
        best_score, best_threshold = -1.0, 0.5
        for threshold in np.linspace(0.05, 0.95, 91):
            score = float(
                f1_score(y_true, (y_prob >= threshold).astype(np.int32), zero_division=0)
            )
            # Prefer the higher threshold on ties to avoid unnecessary positives.
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and threshold > best_threshold
            ):
                best_score, best_threshold = score, float(threshold)
        thresholds[label] = round(best_threshold, 3)
    return thresholds


def metrics_from_arrays(truth, probabilities, known, loss, thresholds):
    predicted = np.zeros_like(truth, dtype=np.int32)
    per_label, diagnostics, f1_values = {}, {}, []
    for index, label in enumerate(LABELS):
        threshold = float(thresholds.get(label, 0.5))
        predicted[:, index] = (probabilities[:, index] >= threshold).astype(np.int32)
        selected = known[:, index] > 0
        y_true = truth[selected, index]
        y_pred = predicted[selected, index]
        y_prob = probabilities[selected, index]
        score = float(f1_score(y_true, y_pred, zero_division=0))
        per_label[label] = score
        if len(np.unique(y_true)) > 1:
            f1_values.append(score)
        diagnostics[label] = {
            "known_rows": int(len(y_true)),
            "actual_positive_rate": round(float(y_true.mean()), 4) if len(y_true) else 0.0,
            "predicted_positive_rate": round(float(y_pred.mean()), 4) if len(y_pred) else 0.0,
            "mean_probability": round(float(y_prob.mean()), 4) if len(y_prob) else 0.0,
            "max_probability": round(float(y_prob.max()), 4) if len(y_prob) else 0.0,
            "threshold": threshold,
        }
    return {
        "loss": float(loss),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_label_f1": per_label,
        "class_diagnostics": diagnostics,
        "evaluated_rows": int(len(truth)),
        "thresholds": {label: float(thresholds.get(label, 0.5)) for label in LABELS},
    }


def evaluate(model, loader, device, pos_weight, thresholds):
    truth, probabilities, known, loss = collect_predictions(
        model, loader, device, pos_weight
    )
    return metrics_from_arrays(truth, probabilities, known, loss, thresholds)


def load_source_split(data_dir, brighter_name, hinemo_name, hinemo_limit, seed):
    brighter = read_jsonl(data_dir / brighter_name)
    hinemo = deterministic_subset(read_jsonl(data_dir / hinemo_name), hinemo_limit, seed)
    return brighter + hinemo, len(brighter), len(hinemo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/text_emotion/prepared")
    parser.add_argument("--output-dir", default="checkpoints/text_emotion_muril")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-hinemo-train", type=int, default=1200)
    parser.add_argument("--max-hinemo-eval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    train_rows, brighter_train_n, hinemo_train_n = load_source_split(
        data_dir, "brighter_hi_train.jsonl", "hinemo_train.jsonl",
        args.max_hinemo_train, args.seed,
    )
    val_rows, brighter_val_n, hinemo_val_n = load_source_split(
        data_dir, "brighter_hi_dev.jsonl", "hinemo_val.jsonl",
        args.max_hinemo_eval, args.seed + 1,
    )
    test_rows, brighter_test_n, hinemo_test_n = load_source_split(
        data_dir, "brighter_hi_test.jsonl", "hinemo_test.jsonl",
        args.max_hinemo_eval, args.seed + 2,
    )

    print(
        f"[TextEmotion] Selected rows: train={len(train_rows)} "
        f"(BRIGHTER {brighter_train_n} + HinEmo {hinemo_train_n}), "
        f"validation={len(val_rows)} (BRIGHTER {brighter_val_n} + HinEmo {hinemo_val_n}), "
        f"test={len(test_rows)} (BRIGHTER {brighter_test_n} + HinEmo {hinemo_test_n})",
        flush=True,
    )
    print(
        f"[TextEmotion] Device={device}; frozen MuRIL encoder; weighted classifier head; "
        f"batch={args.batch_size}; max_length={args.max_length}; epochs={args.epochs}",
        flush=True,
    )
    pos_weight, class_stats = build_positive_weights(train_rows)
    pos_weight = pos_weight.to(device)
    print("[TextEmotion] Training label balance: " + json.dumps(class_stats), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(
        EmotionDataset(train_rows, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        EmotionDataset(val_rows, tokenizer, args.max_length),
        batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        EmotionDataset(test_rows, tokenizer, args.max_length),
        batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_thresholds = {label: 0.5 for label in LABELS}

    for epoch in range(args.epochs):
        model.train()
        model.base_model.eval()  # Keep frozen encoder dropout disabled.
        running_loss, started = 0.0, time.perf_counter()
        total_batches = len(train_loader)
        print(f"[TextEmotion] Epoch {epoch + 1}/{args.epochs} started ({total_batches} batches).", flush=True)
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            masks = batch.pop("label_mask").to(device)
            weights = batch.pop("source_weight").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = masked_weighted_bce(logits, labels, masks, weights, pos_weight)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            if step % 25 == 0 or step == total_batches:
                elapsed = time.perf_counter() - started
                rate = step / max(elapsed, 0.001)
                remaining = (total_batches - step) / max(rate, 0.001)
                print(
                    f"[TextEmotion] Epoch {epoch + 1}: {step}/{total_batches} batches "
                    f"({step / total_batches:.0%}); loss={running_loss / step:.4f}; "
                    f"elapsed={elapsed / 60:.1f} min; ETA={remaining / 60:.1f} min",
                    flush=True,
                )

        val_truth, val_probabilities, val_known, val_loss = collect_predictions(
            model, val_loader, device, pos_weight
        )
        default_validation = metrics_from_arrays(
            val_truth, val_probabilities, val_known, val_loss,
            {label: 0.5 for label in LABELS},
        )
        epoch_thresholds = calibrate_thresholds(
            val_truth, val_probabilities, val_known
        )
        validation = metrics_from_arrays(
            val_truth, val_probabilities, val_known, val_loss, epoch_thresholds
        )
        print(
            "[TextEmotion] Validation at fixed 0.50: "
            + json.dumps(default_validation), flush=True
        )
        print(
            "[TextEmotion] Validation with calibrated thresholds: "
            + json.dumps(validation), flush=True
        )
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            best_thresholds = epoch_thresholds
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"[TextEmotion] Best checkpoint saved to {output_dir}", flush=True)

    best_model = AutoModelForSequenceClassification.from_pretrained(str(output_dir)).to(device)
    test_metrics = evaluate(best_model, test_loader, device, pos_weight, best_thresholds)
    metadata = {
        "model_type": "muril_text_emotion_sequence_classifier",
        "task_type": "multilabel_six_plus_neutral",
        "backbone": MODEL_NAME,
        "labels": LABELS,
        "class_order": LABELS,
        "languages": ["hi", "hinglish"],
        "max_length": args.max_length,
        "training_method": "frozen_muril_encoder_class_weighted_classifier_head_baseline",
        "training_sources": ["BRIGHTER-Hindi", "HinEmo"],
        "hinemo_labels_are_gpt_generated": True,
        "neutral_policy": "neutral is selected only when no non-neutral threshold is met",
        "thresholds": best_thresholds,
        "threshold_calibration": "per_label_f1_sweep_on_validation_only_0.05_to_0.95",
        "training_positive_weights": class_stats,
        "best_validation_macro_f1": best_f1,
        "validation_metrics_at_calibrated_thresholds": validation,
        "test_metrics": test_metrics,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "test_rows": len(test_rows),
        "seed": args.seed,
        "training_note": "CPU-practical baseline; encoder frozen, classifier head trained only.",
    }
    (output_dir / "speechmind_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "text_emotion_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[TextEmotion] Training complete. Saved checkpoint and held-out metrics.", flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
