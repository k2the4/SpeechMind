"""
CPU-practical WavLM voice-emotion experiment for SpeechMind.

On this 6-core / 16 GB CPU-only PC, we do NOT backpropagate through all WavLM
parameters. We extract frozen WavLM mean+standard-deviation embeddings once,
cache them locally, and train/class-balance a small SVM classifier. Several
linear/RBF settings are selected using validation macro-F1. The held-out
speaker test set is evaluated only after model selection.

This is WavLM feature-based training, not full WavLM fine-tuning. It is the
practical path for this computer. It does not guarantee better scores.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time
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
SAMPLE_RATE = 16000
POOLING = "mean_std"
PREPROCESSING_VERSION = "mono_16khz_trim25_peak_normalize_v1"


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
            f"RAVDESS folder not found: {root}. Copy Actor_01 through Actor_24 into data/ravdess."
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
    """Assign whole actors to splits; the same speaker never crosses splits."""
    speakers = sorted({row["speaker_id"] for row in records})
    if len(speakers) < 5:
        raise ValueError(
            f"Need at least 5 distinct RAVDESS actors for group splits; found {len(speakers)}."
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


def pool_hidden_states(hidden_states):
    """Mean+std pool [frames, hidden_dim] WavLM states into a 2*hidden vector."""
    import numpy as np
    values = np.asarray(hidden_states, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Expected non-empty [frames, hidden_dim] WavLM states.")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32, copy=False)


def _cache_fingerprint(records):
    digest = hashlib.sha256()
    digest.update(BASE_MODEL_NAME.encode("utf-8"))
    digest.update(POOLING.encode("utf-8"))
    digest.update(PREPROCESSING_VERSION.encode("utf-8"))
    digest.update(str(SAMPLE_RATE).encode("ascii"))
    for row in records:
        path = Path(row["path"])
        stat = path.stat()
        digest.update(str(path).encode("utf-8", errors="replace"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(row["label"].encode("utf-8"))
        digest.update(row["speaker_id"].encode("utf-8"))
        digest.update(row["split"].encode("ascii"))
    return digest.hexdigest()


def extract_or_load_features(records, cache_path, force_rebuild=False):
    import numpy as np
    import librosa
    import soundfile as sf
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _cache_fingerprint(records)
    if cache_path.is_file() and not force_rebuild:
        try:
            with np.load(cache_path, allow_pickle=False) as saved:
                if str(saved["fingerprint"].item()) == fingerprint:
                    features = saved["features"].astype(np.float32, copy=False)
                    labels = saved["labels"].astype(str)
                    speakers = saved["speakers"].astype(str)
                    splits = saved["splits"].astype(str)
                    if len(features) == len(records):
                        print(f"[Features] Reusing cached WavLM features: {cache_path}")
                        return features, labels, speakers, splits
        except (OSError, ValueError, KeyError):
            print("[Features] Existing cache is invalid; rebuilding it.")

    # Keep resource use predictable and leave headroom for Windows.
    thread_count = max(1, min(4, os.cpu_count() or 2))
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = torch.device("cpu")
    print(f"[Features] Loading frozen {BASE_MODEL_NAME} on CPU...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL_NAME)
    encoder = AutoModel.from_pretrained(BASE_MODEL_NAME).to(device)
    encoder.eval()
    extracted = []
    total = len(records)
    start = time.perf_counter()

    with torch.inference_mode():
        for index, row in enumerate(records, start=1):
            waveform, sample_rate = sf.read(
                row["path"], dtype="float32", always_2d=False
            )
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            if waveform.size == 0 or not np.isfinite(waveform).all():
                raise ValueError(f"Invalid or empty audio file: {row['path']}")
            if sample_rate != SAMPLE_RATE:
                waveform = librosa.resample(
                    waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE
                )
            waveform, _ = librosa.effects.trim(waveform, top_db=25)
            peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
            if peak > 0:
                waveform = waveform / peak
            encoded = feature_extractor(
                waveform.astype(np.float32, copy=False),
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            input_values = encoded["input_values"].to(device)
            hidden = encoder(input_values=input_values).last_hidden_state[0]
            extracted.append(pool_hidden_states(hidden.cpu().numpy()))

            if index == 1 or index % 25 == 0 or index == total:
                elapsed = time.perf_counter() - start
                rate = index / max(elapsed, 1e-6)
                remaining = (total - index) / max(rate, 1e-6)
                print(
                    f"[Features] {index}/{total} clips | "
                    f"{rate:.2f} clips/sec | estimated remaining {remaining / 60:.1f} min"
                )

    features = np.stack(extracted).astype(np.float32, copy=False)
    labels = np.asarray([row["label"] for row in records], dtype=str)
    speakers = np.asarray([row["speaker_id"] for row in records], dtype=str)
    splits = np.asarray([row["split"] for row in records], dtype=str)
    np.savez_compressed(
        cache_path,
        fingerprint=np.asarray(fingerprint),
        features=features,
        labels=labels,
        speakers=speakers,
        splits=splits,
    )
    print(f"[Features] Cached {len(features)} embeddings to {cache_path}")
    del encoder
    return features, labels, speakers, splits


def train_classifier(features, labels, speakers, splits, output_dir, model_dir):
    import json
    import joblib
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from transformers import AutoFeatureExtractor

    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("Train, validation and test splits must all contain audio.")

    X_train, y_train = features[train_mask], labels[train_mask]
    X_val, y_val = features[validation_mask], labels[validation_mask]
    X_test, y_test = features[test_mask], labels[test_mask]

    candidates = []
    for kernel in ("linear", "rbf"):
        for c_value in (0.1, 1.0, 10.0):
            candidate = Pipeline([
                ("scale", StandardScaler()),
                ("classifier", SVC(
                    kernel=kernel,
                    C=c_value,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                )),
            ])
            candidate.fit(X_train, y_train)
            val_predictions = candidate.predict(X_val)
            val_macro_f1 = float(f1_score(
                y_val, val_predictions, labels=LABELS,
                average="macro", zero_division=0,
            ))
            val_accuracy = float(accuracy_score(y_val, val_predictions))
            candidates.append({
                "model": candidate,
                "kernel": kernel,
                "C": c_value,
                "validation_macro_f1": val_macro_f1,
                "validation_accuracy": val_accuracy,
            })
            print(
                f"[Classifier] kernel={kernel:6s} C={c_value:<4} "
                f"validation macro-F1={val_macro_f1:.4f} "
                f"accuracy={val_accuracy:.4f}"
            )

    # Validation macro-F1 selects the classifier; test data is untouched here.
    best = max(
        candidates,
        key=lambda item: (
            item["validation_macro_f1"],
            item["validation_accuracy"],
            -item["C"],
        ),
    )
    best_model = best["model"]
    test_predictions = best_model.predict(X_test)
    test_probability = best_model.predict_proba(X_test)
    class_order = list(best_model.named_steps["classifier"].classes_)
    cm = confusion_matrix(y_test, test_predictions, labels=LABELS)
    report = classification_report(
        y_test, test_predictions, labels=LABELS,
        target_names=LABELS, output_dict=True, zero_division=0,
    )

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / "voice_emotion_head.joblib", compress=3)

    metadata = {
        "model_type": "wavlm_frozen_embedding_svc",
        "backbone": BASE_MODEL_NAME,
        "labels": LABELS,
        "class_order": class_order,
        "sample_rate": SAMPLE_RATE,
        "pooling": POOLING,
        "feature_dimension": int(features.shape[1]),
        "selected_kernel": best["kernel"],
        "selected_C": best["C"],
        "validation_macro_f1": best["validation_macro_f1"],
        "validation_accuracy": best["validation_accuracy"],
        "test_speakers": sorted(set(speakers[test_mask].tolist())),
        "training_speakers": sorted(set(speakers[train_mask].tolist())),
        "validation_speakers": sorted(set(speakers[validation_mask].tolist())),
        "training_method": (
            "Frozen WavLM mean+std embeddings with validation-selected "
            "class-balanced SVM. WavLM weights were not fine-tuned."
        ),
    }
    report_data = {
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "macro_f1": float(f1_score(
            y_test, test_predictions, labels=LABELS,
            average="macro", zero_division=0,
        )),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
        "selected_classifier": {
            "kernel": best["kernel"],
            "C": best["C"],
        },
        "validation": {
            "macro_f1": best["validation_macro_f1"],
            "accuracy": best["validation_accuracy"],
            "candidate_results": [
                {key: item[key] for key in (
                    "kernel", "C", "validation_macro_f1", "validation_accuracy"
                )}
                for item in candidates
            ],
        },
        "test_files": int(test_mask.sum()),
        "note": (
            "Held-out-speaker RAVDESS results. RAVDESS contains acted English "
            "speech; this does not establish accuracy on spontaneous speech "
            "or Indian languages. WavLM embeddings are frozen; only the SVM "
            "classifier was trained."
        ),
    }
    metadata_path = model_dir / "speechmind_metadata.json"
    (model_dir / "feature_extractor_config.json").write_text(
        json.dumps({
            "backbone": BASE_MODEL_NAME,
            "sample_rate": SAMPLE_RATE,
            "pooling": POOLING,
            "preprocessing": PREPROCESSING_VERSION,
        }, indent=2),
        encoding="utf-8",
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_path = output_dir / "test_metrics.json"
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    print("\n[Result] Held-out test evaluation")
    print(f"Accuracy: {report_data['accuracy']:.4f}")
    print(f"Macro-F1: {report_data['macro_f1']:.4f}")
    print(f"Selected: {best['kernel']} SVM, C={best['C']}")
    print(f"Report: {report_path}")
    print(f"Checkpoint: {model_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default="training_outputs/wavlm_cpu")
    parser.add_argument("--model-dir", default="checkpoints/voice_emotion_wavlm_cpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()

    records = assign_speaker_splits(build_manifest(args.data_dir))
    output_dir = Path(args.output_dir)
    manifest_path = write_manifest(
        records, output_dir / "manifest_with_splits.csv"
    )
    summary = summarize(records)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[Dataset] Found {len(records)} labeled RAVDESS speech clips.")
    print(json.dumps(summary, indent=2))
    print(f"[Dataset] Manifest: {manifest_path}")

    if args.prepare_only:
        return

    features, labels, speakers, splits = extract_or_load_features(
        records,
        output_dir / "wavlm_features_cache.npz",
        force_rebuild=args.rebuild_features,
    )
    print(f"[Features] Matrix shape: {features.shape}")
    train_classifier(features, labels, speakers, splits, output_dir, args.model_dir)


if __name__ == "__main__":
    main()
