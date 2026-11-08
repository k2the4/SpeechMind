"""
SpeechMind - Late Fusion Layer

Combines:
    1. Text emotion predictions
    2. Voice emotion predictions

The two branches use different emotion label sets, so their
probability distributions are first mapped into a shared
emotion space.

The text branch is automatically given less weight when
ASR confidence is low, because the transcript may be unreliable.
The voice branch does not depend on the transcript and therefore
receives a larger weight in that situation.
"""


# ============================================================
# CONFIGURATION
# ============================================================

# Shared emotion space used by SpeechMind.
UNIFIED_EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "neutral",
]


# Normal operating weights.
# Text receives slightly more weight because it contains
# semantic/contextual information.
DEFAULT_TEXT_WEIGHT = 0.60
DEFAULT_VOICE_WEIGHT = 0.40


# Below this ASR confidence, reduce reliance on the transcript.
ASR_CONFIDENCE_LOW_THRESHOLD = 0.60


# ============================================================
# TEXT EMOTION MAPPING
# ============================================================

# Maps labels from models/datasets such as:
# GoEmotions, BHAAV, or a custom MuRIL classifier
# into SpeechMind's unified emotion space.

TEXT_TO_UNIFIED = {

    # Joy / positive emotions
    "joy": "joy",
    "happy": "joy",
    "happiness": "joy",
    "excited": "joy",
    "excitement": "joy",
    "love": "joy",
    "gratitude": "joy",
    "grateful": "joy",
    "optimism": "joy",
    "optimistic": "joy",
    "amusement": "joy",
    "celebration": "joy",

    # Sadness
    "sad": "sadness",
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "disappointed": "sadness",
    "remorse": "sadness",

    # Anger
    "angry": "anger",
    "anger": "anger",
    "annoyance": "anger",
    "annoyed": "anger",
    "frustration": "anger",
    "frustrated": "anger",
    "rage": "anger",

    # Fear / nervousness
    "fear": "fear",
    "fearful": "fear",
    "nervous": "fear",
    "nervousness": "fear",
    "anxiety": "fear",
    "worried": "fear",
    "worry": "fear",
    "concern": "fear",
    "confusion": "fear",

    # Surprise
    "surprise": "surprise",
    "surprised": "surprise",

    # Disgust
    "disgust": "disgust",
    "disgusted": "disgust",

    # Neutral
    "neutral": "neutral",
    "calm": "neutral",
}


# ============================================================
# VOICE EMOTION MAPPING
# ============================================================

# Maps common IEMOCAP / RAVDESS / wav2vec2-style labels
# into the same unified emotion space.

VOICE_TO_UNIFIED = {
    "happy": "joy",
    "happiness": "joy",
    "excited": "joy",
    "joy": "joy",

    "sad": "sadness",
    "sadness": "sadness",

    "angry": "anger",
    "anger": "anger",
    "frustrated": "anger",
    "frustration": "anger",

    "fear": "fear",
    "fearful": "fear",

    "surprise": "surprise",
    "surprised": "surprise",

    "disgust": "disgust",
    "disgusted": "disgust",

    "neutral": "neutral",
    "calm": "neutral",
}


# ============================================================
# VALENCE
# ============================================================

# Kept deliberately simple.
#
# Important:
# Surprise is NOT automatically classified as positive because
# surprise can be positive or negative.
#
# Valence can later be improved using the text + audio models.

VALENCE_MAP = {
    "joy": "Positive",
    "sadness": "Negative",
    "anger": "Negative",
    "fear": "Negative",
    "surprise": "Neutral",
    "disgust": "Negative",
    "neutral": "Neutral",
}


# ============================================================
# HELPERS
# ============================================================

def _normalize_distribution(distribution):
    """
    Normalizes a probability distribution so that its values
    sum to 1.

    Example:
        {"happy": 2, "sad": 1}

    becomes approximately:

        {"happy": 0.667, "sad": 0.333}
    """

    if not distribution:
        return {}

    cleaned = {}

    for label, probability in distribution.items():

        try:
            probability = float(probability)
        except (TypeError, ValueError):
            continue

        if probability < 0:
            probability = 0.0

        cleaned[label.lower()] = probability

    total = sum(cleaned.values())

    if total <= 0:
        return {}

    return {
        label: probability / total
        for label, probability in cleaned.items()
    }


def _get_distribution(result, emotion_key):
    """
    Extracts a probability distribution from a model result.

    The preferred format is:

        {
            "emotion": "happy",
            "distribution": {
                "happy": 0.8,
                "neutral": 0.15,
                "sad": 0.05
            }
        }

    If distribution is not available, we create a one-hot
    distribution from the predicted emotion and confidence.
    """

    distribution = result.get("distribution")

    if distribution:
        return _normalize_distribution(distribution)

    emotion = result.get("emotion", "neutral").lower()

    try:
        confidence = float(result.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0

    confidence = max(0.0, min(1.0, confidence))

    # We don't know the probabilities of the other emotions,
    # so use the predicted emotion as the available signal.
    return {
        emotion: confidence
    }


def _map_distribution(distribution, mapping):
    """
    Converts a model-specific emotion distribution into
    SpeechMind's unified emotion space.
    """

    unified = {
        emotion: 0.0
        for emotion in UNIFIED_EMOTIONS
    }

    for label, probability in distribution.items():

        unified_label = mapping.get(
            label.lower(),
            "neutral"
        )

        unified[unified_label] += probability

    # Normalize after merging labels.
    total = sum(unified.values())

    if total > 0:
        unified = {
            label: value / total
            for label, value in unified.items()
        }

    return unified


def _top_emotion(distribution):
    """
    Returns the highest-probability emotion.
    """

    if not distribution:
        return "neutral"

    return max(
        distribution,
        key=distribution.get
    )


# ============================================================
# MAIN FUSION FUNCTION
# ============================================================

def fuse(
    text_result: dict,
    voice_result: dict,
    asr_confidence: float
) -> dict:
    """
    Combine text and voice emotion predictions.

    Parameters
    ----------
    text_result:
        Output from the text emotion classifier.

        Expected format:

        {
            "emotion": "joy",
            "distribution": {
                "joy": 0.80,
                "neutral": 0.15,
                "fear": 0.05
            },
            "confidence": 0.80
        }

    voice_result:
        Output from the voice emotion recognizer.

        Expected format:

        {
            "emotion": "happy",
            "distribution": {
                "happy": 0.75,
                "neutral": 0.20,
                "sad": 0.05
            },
            "confidence": 0.75,
            "arousal": "High"
        }

    asr_confidence:
        Confidence of the speech-to-text system.

    Returns
    -------
    dict
        Result containing the exact fields expected by app.py.
    """

    # --------------------------------------------------------
    # 1. Clean ASR confidence
    # --------------------------------------------------------

    try:
        asr_confidence = float(asr_confidence)
    except (TypeError, ValueError):
        asr_confidence = 0.0

    asr_confidence = max(
        0.0,
        min(1.0, asr_confidence)
    )


    # --------------------------------------------------------
    # 2. Obtain model distributions
    # --------------------------------------------------------

    text_distribution = _get_distribution(
        text_result,
        "emotion"
    )

    voice_distribution = _get_distribution(
        voice_result,
        "emotion"
    )


    # --------------------------------------------------------
    # 3. Convert both branches to unified emotion space
    # --------------------------------------------------------

    text_unified = _map_distribution(
        text_distribution,
        TEXT_TO_UNIFIED
    )

    voice_unified = _map_distribution(
        voice_distribution,
        VOICE_TO_UNIFIED
    )


    # --------------------------------------------------------
    # 4. Determine fusion weights
    # --------------------------------------------------------

    low_confidence = (
        asr_confidence < ASR_CONFIDENCE_LOW_THRESHOLD
    )

    if low_confidence:

        # The weaker the transcript confidence,
        # the less we trust the text branch.

        text_weight = (
            DEFAULT_TEXT_WEIGHT * asr_confidence
        )

        voice_weight = 1.0 - text_weight

    else:

        text_weight = DEFAULT_TEXT_WEIGHT
        voice_weight = DEFAULT_VOICE_WEIGHT


    # --------------------------------------------------------
    # 5. Fuse probability distributions
    # --------------------------------------------------------

    fused_scores = {
        emotion: 0.0
        for emotion in UNIFIED_EMOTIONS
    }

    for emotion in UNIFIED_EMOTIONS:

        fused_scores[emotion] = (
            text_weight * text_unified.get(emotion, 0.0)
            +
            voice_weight * voice_unified.get(emotion, 0.0)
        )


    # --------------------------------------------------------
    # 6. Determine final emotion
    # --------------------------------------------------------

    overall_emotion = _top_emotion(
        fused_scores
    )

    overall_confidence = fused_scores[
        overall_emotion
    ]


    # --------------------------------------------------------
    # 7. Extract individual branch emotions
    # --------------------------------------------------------

    text_emotion = text_result.get(
        "emotion",
        "neutral"
    )

    voice_emotion = voice_result.get(
        "emotion",
        "neutral"
    )


    # --------------------------------------------------------
    # 8. Valence
    # --------------------------------------------------------

    valence = VALENCE_MAP.get(
        overall_emotion,
        "Neutral"
    )


    # --------------------------------------------------------
    # 9. Arousal
    # --------------------------------------------------------

    # The initial implementation uses the voice model's
    # arousal estimate because arousal is strongly associated
    # with acoustic characteristics.

    arousal = voice_result.get(
        "arousal",
        "Unknown"
    )


    # --------------------------------------------------------
    # 10. Human-readable interpretation
    # --------------------------------------------------------

    interpretation_map = {

        "joy":
            "Positive and celebratory emotional expression",

        "sadness":
            "Sad or subdued emotional expression",

        "anger":
            "Angry or frustrated emotional expression",

        "fear":
            "Concerned, nervous, or fear-like emotional expression",

        "surprise":
            "Surprised or unexpected emotional expression",

        "disgust":
            "Disapproving or disgust-like emotional expression",

        "neutral":
            "Neutral or emotionally subdued expression",
    }

    overall_interpretation = interpretation_map.get(
        overall_emotion,
        "Neutral emotional expression"
    )


    # --------------------------------------------------------
    # 11. Return final result
    # --------------------------------------------------------

    return {

        # Fields directly consumed by app.py
        "text_emotion": text_emotion,
        "voice_emotion": voice_emotion,

        "valence": valence,

        "arousal": arousal,

        "confidence": round(
            overall_confidence,
            3
        ),

        "overall_interpretation":
            overall_interpretation,

        "low_confidence_warning":
            low_confidence,

        # Useful debugging information
        "overall_emotion":
            overall_emotion,

        "fusion_weights": {
            "text": round(text_weight, 3),
            "voice": round(voice_weight, 3),
        },

        "asr_confidence":
            round(asr_confidence, 3),

        "text_distribution":
            text_unified,

        "voice_distribution":
            voice_unified,

        "fused_distribution":
            fused_scores,
    }