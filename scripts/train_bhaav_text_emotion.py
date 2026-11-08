"""
BHAAV-to-MuRIL training pipeline for SpeechMind.

BHAAV is Hindi story-text emotion data. This script trains a five-class text
baseline only; it does not claim conversational-domain or audio-emotion accuracy.
It accepts the original Zenodo ZIP or an extracted CSV/XLSX/TSV/JSON file.
"""
from __future__ import annotations

import io
import json
import random
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL_NAME = "google/muril-base-cased"
DEFAULT_OUTPUT_DIR = Path("checkpoints/text_emotion_muril_bhaav")
LABELS = ["anger", "joy", "suspense", "sadness", "neutral"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

LABEL_ALIASES = {
    "anger": "anger", "angry": "anger",
    "joy": "joy", "happy": "joy", "happiness": "joy",
    "suspense": "suspense",
    "sad": "sadness", "sadness": "sadness",
    "neutral": "neutral",
}
TEXT_COLUMN_NAMES = {
    "sentence", "text", "utterance", "sentence_text", "story_sentence",
    "hindi_sentence", "content", "statement",
}
LABEL_COLUMN_NAMES = {
    "emotion", "emotion_label", "label", "class", "category", "emotion_category",
}
GROUP_COLUMN_NAMES = {
    "story_id", "story_name", "story_title", "document_id", "source_id",
    "story", "storyid",
}


def _normalise_column(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _normalise_label(value):
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    key = re.sub(r"[^a-z]+", "", key)
    return LABEL_ALIASES.get(key)


def _read_table_bytes(name, data):
    suffix = Path(name).suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        separator = "\t" if suffix == ".tsv" else None
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return [pd.read_csv(io.BytesIO(data), sep=separator, engine="python", encoding=encoding)]
            except Exception as error:
                last_error = error
        raise ValueError(f"Could not read {name}: {last_error}")
    if suffix in {".xlsx", ".xls"}:
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        except ImportError as error:
            # Install the one optional workbook reader through the active
            # Streamlit/Python environment; no terminal step is required.
            message = str(error).lower()
            if "openpyxl" not in message:
                raise
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "openpyxl"]
            )
            import openpyxl  # noqa: F401
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        return list(sheets.values())
    if suffix == ".json":
        try:
            payload = json.loads(data.decode("utf-8-sig"))
            if isinstance(payload, list):
                return [pd.DataFrame(payload)]
            if isinstance(payload, dict):
                return [pd.DataFrame(payload)]
        except Exception as error:
            raise ValueError(f"Could not read {name}: {error}") from error
    if suffix == ".jsonl":
        return [pd.read_json(io.BytesIO(data), lines=True)]
    return []


def _candidate_tables(source_path):
    path = Path(source_path)
    if path.is_dir():
        for file_path in path.rglob("*"):
            if file_path.is_file():
                yield from (
                    (str(file_path), table)
                    for table in _read_table_bytes(file_path.name, file_path.read_bytes())
                )
        return

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if item.is_dir() or item.file_size > 100_000_000:
                    continue
                if Path(item.filename).suffix.lower() not in {
                    ".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".jsonl"
                }:
                    continue
                data = archive.read(item)
                for table in _read_table_bytes(item.filename, data):
                    yield item.filename, table
        return

    yield str(path), from_single_file(path)


def from_single_file(path):
    return _read_table_bytes(path.name, path.read_bytes())[0]


def _infer_columns(frame):
    columns = {_normalise_column(column): column for column in frame.columns}
    label_col = next(
        (columns[name] for name in LABEL_COLUMN_NAMES if name in columns),
        None,
    )
    text_col = next(
        (columns[name] for name in TEXT_COLUMN_NAMES if name in columns),
        None,
    )

    if label_col is None:
        for normalised, original in columns.items():
            mapped = frame[original].map(_normalise_label)
            valid_ratio = mapped.notna().mean() if len(mapped) else 0
            if valid_ratio >= 0.70 and mapped.nunique() >= 3:
                label_col = original
                break

    if text_col is None:
        candidates = []
        for original in frame.columns:
            if original == label_col:
                continue
            series = frame[original].dropna().astype(str)
            if len(series) and (series.str.len() >= 4).mean() >= 0.50:
                candidates.append((float(series.str.len().median()), original))
        if candidates:
            text_col = max(candidates)[1]

    group_col = next(
        (columns[name] for name in GROUP_COLUMN_NAMES if name in columns),
        None,
    )
    return text_col, label_col, group_col


def load_bhaav_dataframe(source_path):
    candidates = []
    errors = []
    for source_name, frame in _candidate_tables(source_path):
        if frame is None or frame.empty or len(frame.columns) < 2:
            continue
        text_col, label_col, group_col = _infer_columns(frame)
        if text_col is None or label_col is None:
            errors.append(f"{source_name}: could not identify text/label columns ({list(frame.columns)})")
            continue

        clean = pd.DataFrame({
            "text": frame[text_col].astype(str).str.strip(),
            "label": frame[label_col].map(_normalise_label),
        })
        if group_col is not None:
            raw_groups = frame[group_col]
            clean["group"] = raw_groups.where(raw_groups.notna(), None)
            clean["group"] = clean["group"].map(
                lambda value: str(value).strip()
                if value is not None and str(value).strip().lower() not in {"", "nan", "none"}
                else None
            )
        else:
            clean["group"] = None

        clean = clean[
            clean["text"].notna()
            & clean["text"].ne("")
            & clean["label"].notna()
            & clean["text"].str.lower().ne("nan")
        ].copy()
        if len(clean):
            candidates.append((len(clean), source_name, clean))

    if not candidates:
        detail = "\n".join(errors[:8])
        raise ValueError(
            "No supported BHAAV table with recognizable Hindi text and emotion columns was found. "
            "Expected CSV/XLSX/TSV/JSON with sentence/text and emotion/label columns. "
            + detail
        )

    _, source_name, clean = max(candidates, key=lambda item: item[0])
    clean["text_key"] = clean["text"].str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    # Remove exact duplicates. Conflicting duplicate annotations are excluded.
    conflicts = clean.groupby("text_key")["label"].nunique()
    conflicting_keys = set(conflicts[conflicts > 1].index)
    clean = clean[~clean["text_key"].isin(conflicting_keys)]
    clean = clean.drop_duplicates("text_key").reset_index(drop=True)
    clean = clean.drop(columns=["text_key"])

    missing = set(LABELS) - set(clean["label"].unique())
    if missing:
        raise ValueError(f"BHAAV table is missing expected classes: {sorted(missing)}")
    return clean, source_name


def _split_dataframe(frame, seed=42):
    groups = frame["group"]
    use_groups = groups.notna().all() and groups.nunique() >= 5

    if use_groups:
        group_values = groups.astype(str).to_numpy()
        labels = frame["label"].to_numpy()
        for attempt_seed in range(seed, seed + 40):
            first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=attempt_seed)
            train_val_idx, test_idx = next(first.split(frame, labels, groups=group_values))
            second = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=attempt_seed + 1)
            train_sub_idx, val_sub_idx = next(
                second.split(
                    frame.iloc[train_val_idx],
                    labels[train_val_idx],
                    groups=group_values[train_val_idx],
                )
            )
            train_idx = train_val_idx[train_sub_idx]
            val_idx = train_val_idx[val_sub_idx]
            partitions = [frame.iloc[index]["label"].nunique() for index in (train_idx, val_idx, test_idx)]
            if all(value == len(LABELS) for value in partitions):
                return (
                    frame.iloc[train_idx].reset_index(drop=True),
                    frame.iloc[val_idx].reset_index(drop=True),
                    frame.iloc[test_idx].reset_index(drop=True),
                    True,
                )

    if use_groups:
        raise ValueError(
            "A story/source ID column was found, but a complete five-class "
            "speaker/story-disjoint split could not be formed. Refusing to fall "
            "back to sentence-level random splitting because that could leak stories."
        )

    train_val, test = train_test_split(
        frame, test_size=0.20, random_state=seed, stratify=frame["label"]
    )
    train, validation = train_test_split(
        train_val, test_size=0.125, random_state=seed, stratify=train_val["label"]
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True), False


def _encode_frame(tokenizer, frame, max_length):
    encoded = tokenizer(
        frame["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(
        [LABEL_TO_ID[label] for label in frame["label"]],
        dtype=torch.long,
    )
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)


def _evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.inference_mode():
        for input_ids, attention_mask, labels in loader:
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            predictions = output.logits.argmax(dim=-1).cpu().tolist()
            all_pred.extend(predictions)
            all_true.extend(labels.tolist())

    report = classification_report(
        all_true,
        all_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_f1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            all_true, all_pred, labels=list(range(len(LABELS)))
        ).tolist(),
    }


def train_bhaav_model(
    source_path,
    output_dir=DEFAULT_OUTPUT_DIR,
    progress_callback=None,
    max_epochs=None,
):
    def report(message):
        print(message)
        if progress_callback:
            progress_callback(message)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    frame, source_name = load_bhaav_dataframe(source_path)
    train_frame, val_frame, test_frame, group_split = _split_dataframe(frame)
    report(
        f"Loaded {len(frame):,} deduplicated BHAAV sentences from {source_name}. "
        f"Split: {len(train_frame):,} train / {len(val_frame):,} validation / "
        f"{len(test_frame):,} test."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    batch_size = 8 if is_cuda else 2
    epochs = max_epochs or (4 if is_cuda else 2)
    max_length = 128 if is_cuda else 96
    unfreeze_last_layers = 2 if is_cuda else 0

    report(
        f"Training on {device.type.upper()}: batch size {batch_size}, {epochs} epochs, "
        f"unfreezing {unfreeze_last_layers} final MuRIL encoder layers."
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    id2label = {index: label for index, label in enumerate(LABELS)}
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # A light CPU-friendly linear-probe baseline; GPU training fine-tunes the
    # final two encoder blocks in addition to the classification head.
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    if unfreeze_last_layers:
        encoder_layers = model.base_model.encoder.layer
        for layer in encoder_layers[-unfreeze_last_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        pooler = getattr(model.base_model, "pooler", None)
        if pooler is not None:
            for parameter in pooler.parameters():
                parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    train_data = _encode_frame(tokenizer, train_frame, max_length)
    val_data = _encode_frame(tokenizer, val_frame, max_length)
    test_data = _encode_frame(tokenizer, test_frame, max_length)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    class_counts = np.bincount(
        [LABEL_TO_ID[label] for label in train_frame["label"]],
        minlength=len(LABELS),
    )
    class_weights = len(train_frame) / (
        len(LABELS) * np.maximum(class_counts, 1)
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-5, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = max(1, int(total_steps * 0.10))

    def learning_rate_scale(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return max(
            0.0,
            float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate_scale
    )

    best_macro_f1 = -1.0
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_index, (input_ids, attention_mask, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            loss = loss_fn(output.logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())

            if batch_index % 50 == 0:
                report(
                    f"Epoch {epoch + 1}/{epochs}: batch {batch_index}/{len(train_loader)}, "
                    f"average loss {running_loss / batch_index:.4f}"
                )

        val_metrics = _evaluate(model, val_loader, device)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, len(train_loader)),
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_record)
        report(
            f"Epoch {epoch + 1}: validation accuracy {val_metrics['accuracy']:.3f}, "
            f"macro-F1 {val_metrics['macro_f1']:.3f}."
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 2:
                report("Early stopping: validation macro-F1 did not improve for two epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)

    test_metrics = _evaluate(model, test_loader, device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    training_report = {
        "dataset": "BHAAV",
        "source_file": source_name,
        "domain": "Hindi story sentences",
        "domain_warning": (
            "This checkpoint is a Hindi story-text baseline. Its scores do not establish "
            "accuracy on spontaneous speech transcripts or Hinglish conversations."
        ),
        "num_examples_after_deduplication": len(frame),
        "split": {
            "train": len(train_frame),
            "validation": len(val_frame),
            "test": len(test_frame),
            "group_disjoint": group_split,
        },
        "labels": LABELS,
        "device": str(device),
        "batch_size": batch_size,
        "max_length": max_length,
        "epochs_completed": len(history),
        "unfrozen_encoder_layers": unfreeze_last_layers,
        "history": history,
        "best_validation_macro_f1": best_macro_f1,
        "test_metrics": test_metrics,
        "confidence_calibrated": False,
    }
    (output_path / "training_report.json").write_text(
        json.dumps(training_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report(
        f"Training complete. Held-out BHAAV text test accuracy: "
        f"{test_metrics['accuracy']:.3f}; macro-F1: {test_metrics['macro_f1']:.3f}."
    )
    return training_report
