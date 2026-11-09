"""Transparent, heuristic quality indicators for OpenAI Whisper segments.

These values are not calibrated probabilities of transcript correctness.
"""
from math import exp
from statistics import fmean


def estimate_whisper_quality(segments, transcript):
    """Return a conservative token-quality heuristic and its source diagnostics."""
    usable = list(segments or [])
    if not transcript or not str(transcript).strip() or not usable:
        return {
            "quality_score": 0.0,
            "avg_logprob": None,
            "avg_no_speech_prob": None,
            "avg_compression_ratio": None,
        }

    logprobs = [
        float(item["avg_logprob"])
        for item in usable
        if item.get("avg_logprob") is not None
    ]
    no_speech = [
        float(item["no_speech_prob"])
        for item in usable
        if item.get("no_speech_prob") is not None
    ]
    compression = [
        float(item["compression_ratio"])
        for item in usable
        if item.get("compression_ratio") is not None
    ]

    if not logprobs:
        score = 0.0
        mean_logprob = None
    else:
        mean_logprob = fmean(logprobs)
        # exp(mean log probability) is a geometric-mean token score, not
        # the probability that the complete transcript is correct.
        score = exp(max(-5.0, min(0.0, mean_logprob)))

    mean_no_speech = fmean(no_speech) if no_speech else 0.0
    mean_compression = fmean(compression) if compression else 1.0

    score *= 1.0 - max(0.0, min(0.95, mean_no_speech))
    if mean_compression > 2.4:
        score *= max(0.25, 2.4 / mean_compression)

    return {
        "quality_score": round(max(0.0, min(1.0, score)), 3),
        "avg_logprob": round(mean_logprob, 4) if mean_logprob is not None else None,
        "avg_no_speech_prob": round(mean_no_speech, 4),
        "avg_compression_ratio": round(mean_compression, 4),
    }
