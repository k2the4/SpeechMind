"""Pure formatting helpers for SpeechMind's six-signal plus neutral model."""
from __future__ import annotations

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def format_multilabel_scores(scores: dict, thresholds: dict | None = None) -> dict:
    """Turn independent six-way probabilities into SpeechMind's seven-way view.

    Secondary labels remain visible. Neutral is selected only when no
    non-neutral signal reaches its configured decision threshold.
    """
    thresholds = thresholds or {}
    clean = {}
    for label in EMOTION_LABELS:
        try:
            value = float(scores.get(label, 0.0))
            clean[label] = min(1.0, max(0.0, value))
        except (TypeError, ValueError):
            clean[label] = 0.0

    active = [
        label for label in EMOTION_LABELS
        if clean[label] >= float(thresholds.get(label, 0.5))
    ]
    if not active:
        return {
            "emotion": "neutral",
            "confidence": max(0.0, 1.0 - max(clean.values(), default=0.0)),
            "distribution": {"neutral": 1.0},
            "secondary_emotions": [],
            "multi_label_scores": clean,
        }

    neutral_score = 1.0
    for value in clean.values():
        neutral_score *= 1.0 - value

    raw = {label: clean[label] for label in active}
    raw["neutral"] = neutral_score
    total = sum(raw.values())
    distribution = (
        {label: value / total for label, value in raw.items()}
        if total > 0 else {"neutral": 1.0}
    )
    primary = max(distribution, key=distribution.get)
    return {
        "emotion": primary,
        "confidence": distribution[primary],
        "distribution": distribution,
        "secondary_emotions": sorted(
            [label for label in active if label != primary],
            key=lambda label: clean[label],
            reverse=True,
        ),
        "multi_label_scores": clean,
    }
