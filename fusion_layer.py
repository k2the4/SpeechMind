"""
SpeechMind reliability-aware late fusion.

Text and voice are independent observations. Unknown ASR quality is not
silently interpreted as low quality, and an ambiguous voice distribution is
not treated as a contradictory emotion label.
"""

UNIFIED_EMOTIONS = [
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"
]
DEFAULT_TEXT_WEIGHT = 0.60
DEFAULT_VOICE_WEIGHT = 0.40
UNKNOWN_ASR_TEXT_WEIGHT = 0.50
ASR_CONFIDENCE_LOW_THRESHOLD = 0.60
VOICE_AMBIGUITY_MARGIN = 0.05
FUSED_CONFIDENCE_MARGIN_WARNING = 0.15

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


def _empty_result(text_distribution, voice_distribution, uncertainty_reasons,
                  text_available, voice_available, voice_emotion="unavailable",
                  branch_comparison="Insufficient reliable evidence; no combined emotion assigned."):
    return {
        "text_emotion": _top_emotion(text_distribution) or "unavailable",
        "voice_emotion": voice_emotion,
        "valence": "Unknown",
        "arousal": "Not assessed",
        "confidence": None,
        "confidence_margin": 0.0,
        "confidence_is_calibrated": False,
        "overall_interpretation": "Insufficient reliable evidence. No overall emotion has been assigned.",
        "low_confidence_warning": True,
        "overall_emotion": "unresolved",
        "fusion_weights": {"text": 0.0, "voice": 0.0},
        "text_distribution": text_distribution,
        "voice_distribution": voice_distribution,
        "fused_distribution": {},
        "text_branch_available": text_available,
        "voice_branch_available": voice_available,
        "branch_agreement": None,
        "branch_disagreement": False,
        "branch_comparison": branch_comparison,
        "interpretation_status": "insufficient_evidence",
        "uncertainty_reasons": uncertainty_reasons,
        "voice_label_ambiguous": voice_emotion == "uncertain",
        "arousal_available": False,
    }


def fuse(text_result: dict, voice_result: dict, asr_confidence: float) -> dict:
    """Fuse reliable semantic and vocal distributions.

    ASR confidence is an uncalibrated quality indicator. When it is missing,
    the branches receive equal weight rather than assuming that missing means
    poor. An ambiguous voice top-label is excluded from fusion; if text is
    usable it becomes a text-only interpretation, otherwise the result abstains.
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
    text_emotion = _top_emotion(text_distribution)
    voice_ranked = sorted(
        voice_distribution.items(), key=lambda item: item[1], reverse=True
    )
    voice_top_two_margin = (
        voice_ranked[0][1] - voice_ranked[1][1]
        if len(voice_ranked) > 1 else 1.0
    )
    voice_label_ambiguous = bool(
        voice_available and voice_top_two_margin < VOICE_AMBIGUITY_MARGIN
    )
    voice_emotion = (
        "uncertain" if voice_label_ambiguous
        else (_top_emotion(voice_distribution) or "unavailable")
    )
    reliable_voice_available = voice_available and not voice_label_ambiguous

    uncertainty_reasons = []
    if not quality_available:
        uncertainty_reasons.append("ASR quality indicator unavailable")
    elif asr_confidence < ASR_CONFIDENCE_LOW_THRESHOLD:
        uncertainty_reasons.append("ASR quality indicator is low")
    if not text_available:
        uncertainty_reasons.append("Text-emotion branch unavailable")
    if not voice_available:
        uncertainty_reasons.append("Voice-emotion branch unavailable")
    if voice_label_ambiguous:
        uncertainty_reasons.append(
            "Voice model top-two scores are too close to support a stable label"
        )

    if not text_available and not reliable_voice_available:
        if voice_available and not voice_label_ambiguous:
            uncertainty_reasons.append("No usable text or voice emotion evidence")
        return _empty_result(
            text_distribution, voice_distribution, uncertainty_reasons,
            text_available, voice_available, voice_emotion,
        )

    if text_available and reliable_voice_available:
        if not quality_available:
            text_weight = UNKNOWN_ASR_TEXT_WEIGHT
            voice_weight = 1.0 - text_weight
        elif asr_confidence < ASR_CONFIDENCE_LOW_THRESHOLD:
            text_weight = DEFAULT_TEXT_WEIGHT * asr_confidence
            voice_weight = 1.0 - text_weight
        else:
            text_weight, voice_weight = DEFAULT_TEXT_WEIGHT, DEFAULT_VOICE_WEIGHT
    elif reliable_voice_available:
        text_weight, voice_weight = 0.0, 1.0
    else:
        # A weak/ambiguous voice label must not dilute usable text evidence.
        text_weight, voice_weight = 1.0, 0.0

    fused_scores = {
        label: text_weight * text_distribution.get(label, 0.0)
        + voice_weight * (voice_distribution.get(label, 0.0)
                          if reliable_voice_available else 0.0)
        for label in UNIFIED_EMOTIONS
    }
    overall_emotion = _top_emotion(fused_scores)
    ranked_scores = sorted(fused_scores.values(), reverse=True)
    second_score = ranked_scores[1] if len(ranked_scores) > 1 else 0.0
    confidence_margin = max(
        0.0, fused_scores[overall_emotion] - second_score
    )

    both_reliable = text_available and reliable_voice_available
    branch_agreement = (
        text_emotion == _top_emotion(voice_distribution)
        if both_reliable else None
    )
    branch_disagreement = branch_agreement is False

    if branch_disagreement:
        uncertainty_reasons.append("Text and voice branches disagree")
        branch_comparison = (
            f"Text signals {text_emotion}; vocal signals "
            f"{_top_emotion(voice_distribution)}. These are different observations; "
            "no combined emotional state has been assigned."
        )
        return {
            "text_emotion": text_result.get("emotion", text_emotion),
            "voice_emotion": voice_emotion,
            "valence": "Unknown",
            "arousal": "Not assessed",
            "confidence": None,
            "confidence_margin": round(confidence_margin, 3),
            "confidence_is_calibrated": False,
            "overall_interpretation": (
                "Text and vocal signals disagree. No combined emotion has been assigned."
            ),
            "low_confidence_warning": True,
            "overall_emotion": "unresolved",
            "fusion_weights": {
                "text": round(text_weight, 3), "voice": round(voice_weight, 3)
            },
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
            "voice_label_ambiguous": False,
            "arousal_available": False,
        }

    if confidence_margin < FUSED_CONFIDENCE_MARGIN_WARNING:
        uncertainty_reasons.append("Combined distribution has a narrow top-score margin")

    if both_reliable:
        interpretation_status = "branch_agreement"
        branch_comparison = (
            f"Text and voice both signal {text_emotion}. Agreement is not proof "
            "that the interpretation is correct."
        )
        overall_interpretation = INTERPRETATIONS.get(
            overall_emotion, "Interpretation unavailable"
        )
    elif text_available:
        interpretation_status = "single_branch"
        if voice_label_ambiguous:
            branch_comparison = (
                "Voice analysis was inconclusive; the current interpretation "
                "uses text evidence only."
            )
            overall_interpretation = (
                INTERPRETATIONS.get(overall_emotion, "Interpretation unavailable")
                + " Vocal signals were inconclusive, so this interpretation is based on text only."
            )
        elif voice_available:
            branch_comparison = (
                "Voice analysis was not reliable enough to contribute; "
                "the current interpretation uses text evidence only."
            )
            overall_interpretation = INTERPRETATIONS.get(
                overall_emotion, "Interpretation unavailable"
            )
        else:
            branch_comparison = "Only the text branch contributed to this interpretation."
            overall_interpretation = INTERPRETATIONS.get(
                overall_emotion, "Interpretation unavailable"
            )
    else:
        interpretation_status = "single_branch"
        branch_comparison = "Only the voice branch contributed to this interpretation."
        overall_interpretation = INTERPRETATIONS.get(
            overall_emotion, "Interpretation unavailable"
        )

    return {
        "text_emotion": text_result.get("emotion", text_emotion) if text_available else "unavailable",
        "voice_emotion": voice_emotion,
        "valence": VALENCE_MAP.get(overall_emotion, "Unknown"),
        "arousal": "Not assessed",
        "confidence": (
            round(fused_scores[overall_emotion], 3)
            if interpretation_status == "branch_agreement" else None
        ),
        "confidence_margin": (
            round(confidence_margin, 3)
            if interpretation_status == "branch_agreement" else None
        ),
        "confidence_is_calibrated": False,
        "overall_interpretation": overall_interpretation,
        "low_confidence_warning": bool(uncertainty_reasons),
        "overall_emotion": overall_emotion,
        "fusion_weights": {
            "text": round(text_weight, 3), "voice": round(voice_weight, 3)
        },
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
        "voice_label_ambiguous": voice_label_ambiguous,
        "arousal_available": False,
    }
