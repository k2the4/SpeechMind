"""
SpeechMind late-fusion layer.

Unavailable or untrained branches are excluded rather than treated as
neutral predictions. The returned top score is an uncalibrated model score,
not a probability that the prediction is correct.
"""

UNIFIED_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "suspense"]
DEFAULT_TEXT_WEIGHT = 0.60
DEFAULT_VOICE_WEIGHT = 0.40
ASR_CONFIDENCE_LOW_THRESHOLD = 0.60

TEXT_TO_UNIFIED = {
    "joy": "joy", "happy": "joy", "happiness": "joy", "excited": "joy",
    "excitement": "joy", "love": "joy", "gratitude": "joy", "grateful": "joy",
    "optimism": "joy", "optimistic": "joy", "amusement": "joy", "celebration": "joy",
    "sad": "sadness", "sadness": "sadness", "grief": "sadness",
    "disappointment": "sadness", "disappointed": "sadness", "remorse": "sadness",
    "angry": "anger", "anger": "anger", "annoyance": "anger", "annoyed": "anger",
    "frustration": "anger", "frustrated": "anger", "rage": "anger",
    "fear": "fear", "fearful": "fear", "nervous": "fear", "nervousness": "fear",
    "anxiety": "fear", "worried": "fear", "worry": "fear", "concern": "fear",
    "confusion": "fear",
    "surprise": "surprise", "surprised": "surprise",
    "disgust": "disgust", "disgusted": "disgust",
    "neutral": "neutral", "calm": "neutral",
    # BHAAV's suspense label is retained explicitly instead of being
    # inaccurately collapsed into fear.
    "suspense": "suspense",
}
VOICE_TO_UNIFIED = {
    "happy": "joy", "happiness": "joy", "excited": "joy", "joy": "joy",
    "sad": "sadness", "sadness": "sadness",
    "angry": "anger", "anger": "anger", "frustrated": "anger", "frustration": "anger",
    "fear": "fear", "fearful": "fear",
    "surprise": "surprise", "surprised": "surprise",
    "disgust": "disgust", "disgusted": "disgust",
    "neutral": "neutral", "calm": "neutral",
}
VALENCE_MAP = {
    "joy": "Positive", "sadness": "Negative", "anger": "Negative",
    "fear": "Negative", "surprise": "Neutral", "disgust": "Negative",
    "neutral": "Neutral",
    "suspense": "Neutral",
}


def _normalize_distribution(distribution):
    if not distribution:
        return {}
    cleaned = {}
    for label, probability in distribution.items():
        try:
            value = max(0.0, float(probability))
        except (TypeError, ValueError):
            continue
        cleaned[str(label).lower()] = value
    total = sum(cleaned.values())
    return {label: value / total for label, value in cleaned.items()} if total > 0 else {}


def _get_distribution(result):
    if not result.get("available", True):
        return {}
    distribution = result.get("distribution")
    if distribution:
        return _normalize_distribution(distribution)
    emotion = str(result.get("emotion", "neutral")).lower()
    try:
        confidence = max(0.0, min(1.0, float(result.get("confidence", 1.0))))
    except (TypeError, ValueError):
        confidence = 1.0
    return {emotion: confidence} if confidence > 0 else {}


def _map_distribution(distribution, mapping):
    unified = {emotion: 0.0 for emotion in UNIFIED_EMOTIONS}
    for label, probability in distribution.items():
        target = mapping.get(label.lower())
        # Unknown labels are ignored, never silently reclassified as neutral.
        if target is not None:
            unified[target] += probability
    total = sum(unified.values())
    return {label: value / total for label, value in unified.items()} if total > 0 else {}


def _top_emotion(distribution):
    return max(distribution, key=distribution.get) if distribution else "neutral"


def fuse(text_result: dict, voice_result: dict, asr_confidence: float) -> dict:
    try:
        asr_confidence = float(asr_confidence)
    except (TypeError, ValueError):
        asr_confidence = 0.0
    asr_confidence = max(0.0, min(1.0, asr_confidence))

    text_available = bool(text_result.get("available", True))
    voice_available = bool(voice_result.get("available", True))
    text_distribution = _map_distribution(_get_distribution(text_result), TEXT_TO_UNIFIED)
    voice_distribution = _map_distribution(_get_distribution(voice_result), VOICE_TO_UNIFIED)

    text_available = text_available and bool(text_distribution)
    voice_available = voice_available and bool(voice_distribution)
    if not text_available and not voice_available:
        return {
            "text_emotion": "unavailable",
            "voice_emotion": "unavailable",
            "valence": "Unknown",
            "arousal": "Unknown",
            "confidence": 0.0,
            "confidence_is_calibrated": False,
            "emotion_analysis_available": False,
            "overall_interpretation": (
                "Emotion analysis is unavailable because no validated model is "
                "currently enabled for this language."
            ),
            "low_confidence_warning": asr_confidence < ASR_CONFIDENCE_LOW_THRESHOLD,
            "overall_emotion": "unavailable",
            "fusion_weights": {"text": 0.0, "voice": 0.0},
            "text_distribution": {},
            "voice_distribution": {},
            "fused_distribution": {},
        }

    if text_available and voice_available:
        if asr_confidence < ASR_CONFIDENCE_LOW_THRESHOLD:
            text_weight = DEFAULT_TEXT_WEIGHT * asr_confidence
            voice_weight = 1.0 - text_weight
        else:
            text_weight, voice_weight = DEFAULT_TEXT_WEIGHT, DEFAULT_VOICE_WEIGHT
    elif voice_available:
        text_weight, voice_weight = 0.0, 1.0
    else:
        text_weight, voice_weight = 1.0, 0.0

    fused_scores = {
        label: text_weight * text_distribution.get(label, 0.0)
        + voice_weight * voice_distribution.get(label, 0.0)
        for label in UNIFIED_EMOTIONS
    }
    overall_emotion = _top_emotion(fused_scores)
    interpretation = {
        "joy": "Positive and celebratory emotional expression",
        "sadness": "Sad or subdued emotional expression",
        "anger": "Angry or frustrated emotional expression",
        "fear": "Concerned, nervous, or fear-like emotional expression",
        "surprise": "Surprised or unexpected emotional expression",
        "disgust": "Disapproving or disgust-like emotional expression",
        "neutral": "Neutral or emotionally subdued expression",
        "suspense": "Suspenseful or anticipatory emotional expression",
    }
    return {
        "text_emotion": text_result.get("emotion", "unavailable") if text_available else "unavailable",
        "voice_emotion": voice_result.get("emotion", "unavailable") if voice_available else "unavailable",
        "valence": VALENCE_MAP.get(overall_emotion, "Neutral"),
        "arousal": voice_result.get("arousal", "Unknown"),
        "confidence": round(fused_scores[overall_emotion], 3),
        "confidence_is_calibrated": False,
        "emotion_analysis_available": True,
        "overall_interpretation": interpretation.get(overall_emotion, "Neutral emotional expression"),
        "low_confidence_warning": asr_confidence < ASR_CONFIDENCE_LOW_THRESHOLD,
        "overall_emotion": overall_emotion,
        "fusion_weights": {"text": round(text_weight, 3), "voice": round(voice_weight, 3)},
        "text_distribution": text_distribution,
        "voice_distribution": voice_distribution,
        "fused_distribution": fused_scores,
    }
