"""
SpeechMind reliability-aware late fusion.

The fusion score is an uncalibrated combination of model distributions.
Branch disagreement is preserved as an explicit output; a weighted winner
must not hide contradictory text and vocal evidence.
"""

UNIFIED_EMOTIONS = [
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"
]
DEFAULT_TEXT_WEIGHT = 0.60
DEFAULT_VOICE_WEIGHT = 0.40
UNKNOWN_ASR_TEXT_WEIGHT = 0.30
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
}
INTERPRETATIONS = {
    "joy": "Positive and celebratory emotional expression",
    "sadness": "Sad or subdued emotional expression",
    "anger": "Angry or frustrated emotional expression",
    "fear": "Concerned, nervous, or fear-like emotional expression",
    "surprise": "Surprised or unexpected emotional expression",
    "disgust": "Disapproving or disgust-like emotional expression",
    "neutral": "Neutral or emotionally subdued expression",
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
        if value > 0:
            key = str(label).lower().strip()
            cleaned[key] = cleaned.get(key, 0.0) + value
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
        if target is not None:
            unified[target] += probability
    total = sum(unified.values())
    return {label: value / total for label, value in unified.items()} if total > 0 else {}


def _top_emotion(distribution):
    return max(distribution, key=distribution.get) if distribution else None


def fuse(text_result: dict, voice_result: dict, asr_confidence: float) -> dict:
    """Fuse available semantic and vocal distributions.

    asr_confidence is a compatibility name for an uncalibrated ASR quality
    indicator. The resulting weights are heuristics until learned/validated
    on representative paired speech-emotion data.
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
    text_available = bool(text_distribution)
    voice_available = bool(voice_distribution)

    if not text_available and not voice_available:
        # Insufficient evidence is a valid product outcome. Never manufacture a
        # neutral label when neither independent branch returned usable output.
        uncertainty_reasons = [
            "Text-emotion branch unavailable",
            "Voice-emotion branch unavailable",
        ]
        if not quality_available:
            uncertainty_reasons.append("ASR quality indicator unavailable")
        return {
            "text_emotion": "unavailable",
            "voice_emotion": "unavailable",
            "valence": "Unknown",
            "arousal": "Not assessed",
            "confidence": 0.0,
            "confidence_margin": 0.0,
            "confidence_is_calibrated": False,
            "overall_interpretation": (
                "Insufficient evidence: neither the text nor voice branch "
                "returned a usable analysis."
            ),
            "low_confidence_warning": True,
            "overall_emotion": "unavailable",
            "fusion_weights": {"text": 0.0, "voice": 0.0},
            "text_distribution": {},
            "voice_distribution": {},
            "fused_distribution": {},
            "text_branch_available": False,
            "voice_branch_available": False,
            "branch_agreement": None,
            "branch_disagreement": False,
            "branch_comparison": (
                "Neither analysis branch returned usable evidence; no combined "
                "emotion has been assigned."
            ),
            "interpretation_status": "insufficient_evidence",
            "uncertainty_reasons": uncertainty_reasons,
            "arousal_available": False,
        }

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
        label: text_weight * text_distribution.get(label, 0.0)
        + voice_weight * voice_distribution.get(label, 0.0)
        for label in UNIFIED_EMOTIONS
    }
    overall_emotion = _top_emotion(fused_scores)
    second_score = sorted(fused_scores.values(), reverse=True)[1]
    confidence_margin = max(0.0, fused_scores[overall_emotion] - second_score)

    text_emotion = _top_emotion(text_distribution)
    voice_emotion = _top_emotion(voice_distribution)
    both_available = text_available and voice_available
    branch_agreement = (
        text_emotion == voice_emotion if both_available else None
    )
    branch_disagreement = branch_agreement is False

    uncertainty_reasons = []
    if not quality_available:
        uncertainty_reasons.append("ASR quality indicator unavailable")
    elif asr_confidence < ASR_CONFIDENCE_LOW_THRESHOLD:
        uncertainty_reasons.append("ASR quality indicator is low")
    if not text_available:
        uncertainty_reasons.append("Text-emotion branch unavailable")
    if not voice_available:
        uncertainty_reasons.append("Voice-emotion branch unavailable")
    if branch_disagreement:
        uncertainty_reasons.append("Text and voice branches disagree")
    if confidence_margin < 0.15:
        uncertainty_reasons.append("Combined distribution has a narrow top-score margin")

    if branch_disagreement:
        interpretation_status = "branch_disagreement"
        branch_comparison = (
            f"Text signals {text_emotion}; vocal signals {voice_emotion}. "
            "These are different observations, so the combined label should be "
            "treated cautiously rather than as a resolved emotional state."
        )

        # Do not manufacture a combined winner from contradictory modalities.
        # Keep the distributions for inspection, but leave the overall label
        # and valence unresolved.
        return {
            "text_emotion": text_result.get("emotion", "unavailable"),
            "voice_emotion": voice_result.get("emotion", "unavailable"),
            "valence": "Unknown",
            "arousal": "Not assessed",
            "confidence": None,
            "confidence_margin": round(confidence_margin, 3),
            "confidence_is_calibrated": False,
            "overall_interpretation": (
                "Text and vocal signals conflict. No combined emotion has been assigned."
            ),
            "low_confidence_warning": True,
            "overall_emotion": "unresolved",
            "fusion_weights": {"text": round(text_weight, 3), "voice": round(voice_weight, 3)},
            "text_distribution": text_distribution,
            "voice_distribution": voice_distribution,
            "fused_distribution": fused_scores,
            "text_branch_available": True,
            "voice_branch_available": True,
            "branch_agreement": False,
            "branch_disagreement": True,
            "branch_comparison": branch_comparison,
            "interpretation_status": "branch_disagreement",
            "uncertainty_reasons": uncertainty_reasons,
            "arousal_available": False,
        }
    elif branch_agreement is True:
        interpretation_status = "branch_agreement"
        branch_comparison = (
            f"Text and voice both signal {text_emotion}. Agreement is not proof "
            "that the interpretation is correct."
        )
    elif text_available or voice_available:
        interpretation_status = "single_branch"
        available_branch = "text" if text_available else "voice"
        branch_comparison = (
            f"Only the {available_branch} branch contributed to this interpretation."
        )
    else:
        interpretation_status = "insufficient_evidence"
        branch_comparison = "Neither analysis branch returned usable evidence."

    return {
        "text_emotion": text_result.get("emotion", "unavailable") if text_available else "unavailable",
        "voice_emotion": voice_result.get("emotion", "unavailable") if voice_available else "unavailable",
        "valence": VALENCE_MAP.get(overall_emotion, "Unknown"),
        # No independently validated acoustic arousal estimator is implemented.
        "arousal": "Not assessed",
        "confidence": round(fused_scores[overall_emotion], 3),
        "confidence_margin": round(confidence_margin, 3),
        "confidence_is_calibrated": False,
        "overall_interpretation": INTERPRETATIONS.get(overall_emotion, "Interpretation unavailable"),
        "low_confidence_warning": bool(uncertainty_reasons),
        "overall_emotion": overall_emotion,
        "fusion_weights": {"text": round(text_weight, 3), "voice": round(voice_weight, 3)},
        "text_distribution": text_distribution,
        "voice_distribution": voice_distribution,
        "fused_distribution": fused_scores,
        "text_branch_available": text_available,
        "voice_branch_available": voice_available,
        "branch_agreement": branch_agreement,
        "branch_disagreement": branch_disagreement,
        "branch_comparison": branch_comparison,
        "interpretation_status": interpretation_status,
        "uncertainty_reasons": uncertainty_reasons,
        "arousal_available": False,
    }
