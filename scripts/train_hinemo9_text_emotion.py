"""Train a Hinglish text-emotion baseline using the human-evaluated HinEmo-9 splits.

The training split contains human-gold and model-derived records. Model-derived
rows receive lower, class-specific sample weights. Dev/test are required to
contain human-gold labels only. This is a text-only baseline, not voice emotion.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import LinearSVC

LABELS = [
    "Love", "Joy", "Anger", "Sadness", "Fear",
    "Surprise", "Nostalgia", "Devotion", "Neutral",
]
LABEL_TO_ID = {label.lower(): label for label in LABELS}

# Estimated label accuracy of the model-derived tier, as reported in the
# HinEmo-9 dataset card. Gold annotations always receive weight 1.0.
LLM_LABEL_QUALITY = {
    "love": 0.59,
    "joy": 0.52,
    "anger": 0.82,
    "sadness": 0.82,
    "fear": 0.80,
    "surprise": 0.92,
    "nostalgia": 0.87,
    "devotion": 0.87,
    "neutral": 0.86,
}
LLM_WEIGHT_SCALE = 0.50


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}") from error
            rows.append(item)
    return rows


def validate_rows(rows: list[dict], split: str, require_gold: bool = False):
    texts, labels, weights, languages = [], [], [], []
    skipped = 0
    for index, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        raw_label = str(row.get("label", "")).strip().lower()
        label = LABEL_TO_ID.get(raw_label)
        source = str(row.get("label_source", "")).strip().lower()
        if not text:
            skipped += 1
            continue
        if label is None:
            raise ValueError(
                f"Unknown emotion label '{raw_label}' in {split} row {index + 1}. "
                "Do not silently drop or remap labels."
            )
        if require_gold and source != "human_gold":
            raise ValueError(
                f"{split} contains a non-gold record at row {index + 1}; "
                "evaluation splits must remain human-annotated."
            )
        if split == "train":
            if source == "human_gold":
                weight = 1.0
            elif source == "llm":
                weight = LLM_WEIGHT_SCALE * LLM_LABEL_QUALITY[raw_label]
            else:
                raise ValueError(
                    f"Unknown label_source '{source}' in training record {index + 1}."
                )
        else:
            weight = 1.0

        texts.append(text)
        labels.append(label)
        weights.append(weight)
        languages.append(str(row.get("language_label", "unknown")).lower())

    if not texts:
        raise ValueError(f"No usable records were found in split '{split}'.")
    return texts, labels, np.asarray(weights, dtype=np.float32), languages, skipped


def build_features():
    # Character n-grams are particularly useful for Romanized Hindi, spelling
    # variation and code-mixed token boundaries. Word n-grams add phrase context.
    return (
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 6),
            min_df=2,
            max_features=350_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            min_df=2,
            max_features=120_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
    )


def metrics_for(y_true, y_pred):
    labels = LABELS
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/hinemo9/data")
    parser.add_argument("--output-dir", default="checkpoints/text_emotion_hinemo9")
    parser.add_argument("--report-dir", default="training_outputs/hinemo9_text_emotion")
    parser.add_argument("--candidates", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--max-iter", type=int, default=4000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    dev_rows = read_jsonl(data_dir / "dev.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")

    x_train, y_train, sample_weights, train_languages, train_skipped = validate_rows(
        train_rows, "train"
    )
    x_dev, y_dev, _, dev_languages, dev_skipped = validate_rows(
        dev_rows, "dev", require_gold=True
    )
    x_test, y_test, _, test_languages, test_skipped = validate_rows(
        test_rows, "test", require_gold=True
    )

    print(
        f"[HinEmo-9] usable train={len(x_train)} (gold and down-weighted model labels), "
        f"dev={len(x_dev)} gold, test={len(x_test)} gold"
    )
    print(
        "[HinEmo-9] train label sources: "
        + json.dumps(dict(Counter(
            str(row.get("label_source", "unknown")).lower()
            for row in train_rows
            if str(row.get("label", "")).strip().lower() in LABEL_TO_ID
        )))
    )

    char_vectorizer, word_vectorizer = build_features()
    print("[Features] Fitting character n-gram TF-IDF...")
    x_train_char = char_vectorizer.fit_transform(x_train)
    x_dev_char = char_vectorizer.transform(x_dev)
    x_test_char = char_vectorizer.transform(x_test)

    print("[Features] Fitting word n-gram TF-IDF...")
    x_train_word = word_vectorizer.fit_transform(x_train)
    x_dev_word = word_vectorizer.transform(x_dev)
    x_test_word = word_vectorizer.transform(x_test)

    x_train_all = hstack([x_train_char, x_train_word], format="csr")
    x_dev_all = hstack([x_dev_char, x_dev_word], format="csr")
    x_test_all = hstack([x_test_char, x_test_word], format="csr")
    print(
        f"[Features] train matrix={x_train_all.shape}; "
        f"features={x_train_all.shape[1]}"
    )

    best_model = None
    best_c = None
    best_dev_f1 = -1.0
    candidate_metrics = []
    for c_value in args.candidates:
        classifier = LinearSVC(
            C=float(c_value),
            class_weight="balanced",
            max_iter=args.max_iter,
            dual=True,
            random_state=42,
        )
        classifier.fit(
            x_train_all, y_train, sample_weight=sample_weights
        )
        dev_predictions = classifier.predict(x_dev_all)
        dev_metrics = metrics_for(y_dev, dev_predictions)
        candidate_metrics.append({
            "C": float(c_value),
            "validation_accuracy": dev_metrics["accuracy"],
            "validation_macro_f1": dev_metrics["macro_f1"],
        })
        print(
            f"[Classifier] LinearSVC C={c_value:g} "
            f"dev accuracy={dev_metrics['accuracy']:.4f}, "
            f"macro-F1={dev_metrics['macro_f1']:.4f}"
        )
        if dev_metrics["macro_f1"] > best_dev_f1:
            best_model = classifier
            best_c = float(c_value)
            best_dev_f1 = dev_metrics["macro_f1"]

    test_predictions = best_model.predict(x_test_all)
    test_metrics = metrics_for(y_test, test_predictions)

    # Evaluate the same gold test set separately by its three language varieties.
    test_by_language = {}
    for language in sorted(set(test_languages)):
        mask = np.asarray([value == language for value in test_languages])
        if not mask.any():
            continue
        test_by_language[language] = metrics_for(
            np.asarray(y_test)[mask].tolist(),
            test_predictions[mask].tolist(),
        )

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "char_vectorizer": char_vectorizer,
            "word_vectorizer": word_vectorizer,
            "classifier": best_model,
            "labels": LABELS,
        },
        output_dir / "text_emotion_head.joblib",
        compress=3,
    )
    metadata = {
        "model_type": "hinemo9_char_word_tfidf_linear_svc",
        "dataset": "HinEmo-9",
        "dataset_repository": "https://github.com/im-rakeshsahani/hinemo9",
        "labels": LABELS,
        "class_order": [str(label) for label in best_model.classes_],
        "languages": ["en", "hi", "hinglish"],
        "language_varieties_in_data": sorted(set(train_languages)),
        "training_method": (
            "Character and word TF-IDF + class-balanced LinearSVC; "
            "human-gold train rows weight 1.0; model-derived train rows "
            "weighted by reported per-class agreement."
        ),
        "selected_C": best_c,
        "validation_macro_f1": best_dev_f1,
        "train_rows": len(x_train),
        "validation_rows": len(x_dev),
        "test_rows": len(x_test),
        "evaluation_label_source": "human_gold_only",
        "training_label_source_counts": dict(Counter(
            str(row.get("label_source", "unknown")).lower()
            for row in train_rows
            if str(row.get("label", "")).strip().lower() in LABEL_TO_ID
        )),
        "limitations": [
            "The source is YouTube comments, not transcribed spontaneous speech.",
            "Most training labels are model-derived and have uneven per-class accuracy.",
            "Use as a Hinglish text baseline; validate on SpeechMind speech transcripts.",
            "The classifier decision scores are not calibrated probabilities.",
        ],
    }
    (output_dir / "speechmind_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    report = {
        "dataset": "HinEmo-9",
        "selected_C": best_c,
        "validation_macro_f1": best_dev_f1,
        "candidate_results": candidate_metrics,
        "test": test_metrics,
        "test_by_language": test_by_language,
        "row_counts": {
            "train_used": len(x_train),
            "dev_gold_used": len(x_dev),
            "test_gold_used": len(x_test),
            "train_skipped": train_skipped,
            "dev_skipped": dev_skipped,
            "test_skipped": test_skipped,
        },
        "warning": (
            "Held-out evaluation is on human-annotated YouTube comments, not "
            "natural spoken conversations. Do not report this as voice or ASR accuracy."
        ),
    }
    (report_dir / "test_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"[Result] HinEmo-9 gold test accuracy={test_metrics['accuracy']:.4f}, "
        f"macro-F1={test_metrics['macro_f1']:.4f}"
    )
    for language, metrics in test_by_language.items():
        print(
            f"[Result] {language}: accuracy={metrics['accuracy']:.4f}, "
            f"macro-F1={metrics['macro_f1']:.4f}"
        )
    print(f"[Result] Checkpoint: {output_dir}")
    print(f"[Result] Metrics: {report_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
