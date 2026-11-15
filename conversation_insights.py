"""Transparent, lightweight conversation signals for the SpeechMind MVP.

This module deliberately uses deterministic text heuristics. It does not claim
to perform generative summarization, verified complaint classification, or
reliable intent detection. Every output is a review cue grounded in transcript
text and should be checked against the original audio.
"""
import re
from collections import Counter

TOPIC_KEYWORDS = {
    "Payments & refunds": ("payment", "paid", "refund", "money", "transaction", "billing", "charge", "paisa", "paise", "payment", "refund"),
    "Orders & delivery": ("order", "delivery", "delivered", "shipment", "courier", "parcel", "late", "dispatch"),
    "Account access": ("account", "login", "log in", "password", "otp", "verification", "blocked", "sign in"),
    "Technical support": ("error", "issue", "problem", "not working", "crash", "failed", "app", "website", "server"),
    "Cancellation & returns": ("cancel", "cancellation", "return", "replace", "replacement", "exchange"),
}
ISSUE_CUES = (
    "problem", "issue", "complaint", "not working", "not received",
    "haven't received", "have not received", "didn't receive", "did not receive",
    "still waiting", "not resolved", "not fixed", "failed", "delay", "late",
    "refund nahi", "nahi aaya", "nahi mila", "abhi tak", "paise wapas",
)
REPEAT_CONTACT_CUES = (
    "called again", "called three times", "called multiple times",
    "contacted before", "already called", "again and again",
    "multiple times", "third time", "pehle bhi", "kai baar",
)
FOLLOW_UP_CUES = (
    "please check", "please resolve", "call me back", "follow up",
    "still waiting", "not resolved", "abhi tak", "nahi mila",
)


def _sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?।])\s+|\n+", text or "") if part.strip()]


def analyze_conversation_text(transcript, translated_text=None):
    """Return conservative, transcript-grounded signals for a human reviewer."""
    original = (transcript or "").strip()
    english = (translated_text or "").strip()
    source = english if english else original
    lowered = source.casefold()
    sentences = _sentences(source)

    # Extractive brief: retain the first two transcript sentences, not an
    # invented summary. Labeling in the UI makes this limitation explicit.
    brief = " ".join(sentences[:2]) if sentences else "No usable transcript is available."
    if len(brief) > 420:
        brief = brief[:417].rsplit(" ", 1)[0] + "..."

    topics = []
    for topic, words in TOPIC_KEYWORDS.items():
        matched = sorted({word for word in words if word in lowered})
        if matched:
            topics.append({"topic": topic, "matched_terms": matched[:4]})

    issue_matches = [cue for cue in ISSUE_CUES if cue in lowered]
    repeat_matches = [cue for cue in REPEAT_CONTACT_CUES if cue in lowered]
    follow_matches = [cue for cue in FOLLOW_UP_CUES if cue in lowered]

    signals = []
    if issue_matches:
        signals.append({
            "label": "Possible unresolved issue",
            "evidence": issue_matches[:4],
            "description": "Issue-related wording appears in the transcript.",
        })
    if repeat_matches:
        signals.append({
            "label": "Possible repeat contact",
            "evidence": repeat_matches[:3],
            "description": "The speaker may be referring to earlier contact attempts.",
        })
    if follow_matches:
        signals.append({
            "label": "Possible follow-up needed",
            "evidence": follow_matches[:4],
            "description": "The transcript contains wording associated with a request or unresolved next step.",
        })

    return {
        "brief": brief,
        "brief_source": "English translation" if english else "original transcript",
        "topics": topics,
        "signals": signals,
        "needs_human_review": bool(signals) or not bool(original),
        "method": "keyword and extractive heuristics",
    }
