"""
SpeechMind multimodal fusion.

Text emotion preserves the nine-label Hinglish taxonomy. Voice emotion retains
the seven categories represented by the current acoustic training data. The
fusion stage does not invent voice evidence for text-only labels such as Love,
Nostalgia or Devotion; it combines available distributions and records the
individual branch outputs.
"""

UNIFIED_EMOTIONS = [
    "love", "joy", "anger", "sadness", "fear",
    "surprise", "nostalgia", "devotion", "disgust", "neutral",
]
DEFAULT_TEXT_WEIGHT = 0.60
DEFAULT_VOICE_WEIGHT = 0.40
UNKNOWN_ASR_TEXT_WEIGHT = 0.30
ASR_CONFIDENCE_LOW_THRESHOLD = 0.60

TEXT_TO_UNIFIED = {
    "love": "love",
    "joy": "joy", "happy": "joy", "happiness": "joy", "excited": "joy",
    "excitement": "joy", "gratitude": "joy", "grateful": "joy",
    "optimism": "joy", "optimistic": "joy", "amusement": "joy", "celebration": "joy",
    "sad": "sadness", "sadness": "sadness", "grief": "sadness",
    "disappointment": "sadness", "disappointed": "sadness", "remorse": "sadness",
    "angry": "anger", "anger": "anger", "annoyance": "anger", "annoyed": "anger",
    "frustration": "anger", "frustrated": "anger", "rage": "anger",
    "fear": "fear", "fearful": "fear", "nervous": "fear", "nervousness": "fear",
    "anxiety": "fear", "worried": "fear", "worry": "fear", "concern": "fear",
    "confusion": "fear",
    "surprise": "surprise", "surprised": "surprise",
    "nostalgia": "nostalgia",
    "devotion": "devotion",
    "disgust": "disgust", "disgusted": "disgust",
    "neutral": "neutral", "calm": "neutral",
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
    "love": "Positive",
    "joy": "Positive",
    "anger": "Negative",
    "sadness": "Negative",
    "fear": "Negative",
    "surprise": "Neutral",
    "nostalgia": "Mixed",
    "devotion": "Positive",
    "disgust": "Negative",
    "neutral": "Neutral",
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
    return (
        {label: value / total for label, value in cleaned.items()}
        if total > 0 else {}
    )


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
        if target is not None:
            unified[target] += probability
    total = sum(unified.values())
    return (
        {label: value / total for label, value in unified.items()}
        if total > 0 else {}
    )


def _top_emotion(distribution):
    return max(distribution, key=distribution.get) if distribution else "neutral"


def fuse(text_result: dict, voice_result: dict, asr_confidence: float) -> dict:
    """Fuse text/voice distributions using the available ASR quality indicator.

    Scores remain uncalibrated. Text-only categories remain available in the
    combined distribution but receive no fabricated vocal probability.
    """
    quality_available = asr_confidence is not None
    if quality_available:
        try:
            asr_confidence = float(asr_confidence)
        except (TypeError, ValueError):
            quality_available = False
    if quality_available:
        asr_confidence = max(0.0, min(1.0, asr_confidence))

    text_distribution = _map_distribution(
        _get_distribution(text_result), TEXT_TO_UNIFIED
    )
    voice_distribution = _map_distribution(
        _get_distribution(voice_result), VOICE_TO_UNIFIED
    )
    text_available = bool(text_result.get("available", True)) and bool(text_distribution)
    voice_available = bool(voice_result.get("available", True)) and bool(voice_distribution)

    if not text_available and not voice_available:
        raise ValueError(
            "Fusion cannot run: both emotion branches are unavailable or returned unmapped labels."
        )

    if text_available and voice_available:
        if not quality_available:
            text_weight = UNKNOWN_ASR_TEXT_WEIGHT
            voice_weight = 1.0 - text_weight
        elif asr_confidence < ASR_CONFIDENCE_LOW_THRESHOLD:
            text_weight = DEFAULT_TEXT_WEIGHT * asr_confidence
            voice_weight = 1.0 - text_weight
        else:
            text_weight, voice_weight = DEFAULT_TEXT_WEIGHT, DEFAULT_VOICE_WEIGHT
    elif voice_available:
        text_weight, voice_weight = 0.0, 1.0
    else:
        text_weight, voice_weight = 1.0, 0.0

    fused_scores = {
        label: (
            text_weight * text_distribution.get(label, 0.0)
            + voice_weight * voice_distribution.get(label, 0.0)
        )
        for label in UNIFIED_EMOTIONS
    }
    overall_emotion = _top_emotion(fused_scores)
    interpretation = {
        "love": "Affectionate or loving emotional expression in the words",
        "joy": "Positive and celebratory emotional expression",
        "anger": "Angry or frustrated emotional expression",
        "sadness": "Sad or hurt emotional expression",
        "fear": "Concerned, nervous or fear-like emotional expression",
        "surprise": "Surprised or unexpected emotional expression",
        "nostalgia": "Bittersweet longing or fond remembrance",
        "devotion": "Reverential or devotional emotional expression",
        "disgust": "Disapproving or disgust-like emotional expression",
        "neutral": "Neutral or emotionally subdued expression",
    }
    return {
        "text_emotion": text_result.get("emotion", "unavailable"),
        "voice_emotion": voice_result.get("emotion", "unavailable"),
        "valence": VALENCE_MAP.get(overall_emotion, "Neutral"),
        "arousal": voice_result.get("arousal", "Unknown"),
        "confidence": round(fused_scores[overall_emotion], 3),
        "confidence_is_calibrated": False,
        "overall_interpretation": interpretation.get(
            overall_emotion, "Neutral emotional expression"
        ),
        "low_confidence_warning": (
            not quality_available
            or asr_confidence < ASR_CONFIDENCE_LOW_THRESHOLD
        ),
        "overall_emotion": overall_emotion,
        "fusion_weights": {
            "text": round(text_weight, 3),
            "voice": round(voice_weight, 3),
        },
        "text_distribution": text_distribution,
        "voice_distribution": voice_distribution,
        "fused_distribution": fused_scores,
        "branch_agreement": (
            "agree"
            if text_available and voice_available
            and text_result.get("emotion") == voice_result.get("emotion")
            else "different_or_unavailable"
        ),
    }
