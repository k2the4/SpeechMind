"""Local smoke test for the trained Hindi/Hinglish text-emotion checkpoint."""
from __future__ import annotations

import json
from models.text_emotion import TextEmotionClassifier


def main() -> int:
    classifier = TextEmotionClassifier()
    if not classifier.available:
        raise SystemExit(
            "FAIL: Hindi/Hinglish text-emotion checkpoint did not load: "
            + str(classifier.load_error)
        )

    samples = [
        ("hi", "मुझे यह बहुत पसंद आया और मैं बहुत खुश हूँ"),
        ("hinglish", "ye result dekh ke main bahut khush hoon"),
        ("hi", "आज का दिन सामान्य था"),
    ]
    print(
        f"[TextEmotionCheck] Loaded {classifier.model_source}; "
        f"languages={sorted(classifier.supported_languages)}; "
        f"max_length={classifier.max_length}"
    )

    for language, text in samples:
        result = classifier.predict(text, language)
        if not result.get("available"):
            raise SystemExit(
                f"FAIL: prediction unavailable for {language}: {result.get('reason')}"
            )
        scores = result.get("multi_label_scores") or {}
        if set(scores) != set(classifier.labels or []):
            raise SystemExit(f"FAIL: unexpected emotion score labels: {sorted(scores)}")
        if any(not 0.0 <= float(value) <= 1.0 for value in scores.values()):
            raise SystemExit("FAIL: emotion score outside [0, 1]")
        print(json.dumps({
            "language": language,
            "emotion": result.get("emotion"),
            "confidence": round(float(result.get("confidence", 0.0)), 4),
            "secondary_emotions": result.get("secondary_emotions", []),
            "scores": {key: round(float(value), 4) for key, value in scores.items()},
        }, ensure_ascii=False))

    print("[TextEmotionCheck] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
