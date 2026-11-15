"""Evidence-grounded customer-support conversation review cues.

This MVP intentionally uses deterministic lexical rules, not an LLM. Findings
are review prompts, never verified complaint states or customer intent. Each
finding carries the sentence(s) that triggered it so a human can check the
recording. Translation is used as a secondary analysis source only when present.
"""
import re
from typing import Dict, Iterable, List, Optional, Tuple


TOPIC_KEYWORDS = {
    "Payments & refunds": (
        "payment", "paid", "refund", "money", "transaction", "billing",
        "charge", "paisa", "paise", "payment failed", "deducted",
        "double charged", "पैसे", "भुगतान", "रिफंड",
    ),
    "Orders & delivery": (
        "order", "delivery", "delivered", "shipment", "courier", "parcel",
        "dispatch", "tracking", "not delivered", "आर्डर", "ऑर्डर", "डिलीवरी",
    ),
    "Account access": (
        "account", "login", "log in", "password", "otp", "verification",
        "blocked", "sign in", "लॉगिन", "पासवर्ड", "ओटीपी",
    ),
    "Technical support": (
        "error", "issue", "problem", "not working", "crash", "failed",
        "app", "website", "server", "bug", "लोड नहीं", "काम नहीं कर",
    ),
    "Cancellation & returns": (
        "cancel", "cancellation", "return", "replace", "replacement",
        "exchange", "वापस", "रिटर्न", "कैंसल",
    ),
    "Service & agent experience": (
        "agent", "executive", "support", "customer care", "rude",
        "helpful", "behaviour", "behavior", "representative",
    ),
}

# Phrases are deliberately explicit to reduce accidental substring matches.
ISSUE_CUES = (
    "problem", "issue", "complaint", "not working", "not received",
    "haven't received", "have not received", "didn't receive",
    "did not receive", "still waiting", "not resolved", "not fixed",
    "not arrived", "has not arrived", "hasn't arrived", "failed",
    "delay", "delayed", "late", "refund nahi", "nahi aaya",
    "nahi mila", "abhi tak", "paise wapas", "काम नहीं कर",
)
RESOLUTION_CUES = (
    "issue is resolved", "problem is resolved", "resolved now",
    "has been resolved", "got the refund", "refund received",
    "delivered now", "working now", "fixed now", "समस्या हल",
)
REPEAT_CONTACT_CUES = (
    "called again", "called three times", "called multiple times",
    "contacted before", "already called", "again and again",
    "multiple times", "third time", "pehle bhi", "पहले भी", "kai baar",
    "कई बार",
)
FOLLOW_UP_CUES = (
    "please check", "please resolve", "call me back", "follow up",
    "follow-up", "update me", "let me know", "still waiting",
    "not resolved", "abhi tak", "nahi mila", "कॉल बैक",
)
ESCALATION_CUES = (
    "escalate", "escalation", "manager", "supervisor", "consumer court",
    "complaint to", "legal action", "social media", "senior team",
    "मैनेजर", "शिकायत",
)
CUSTOMER_IMPACT_CUES = (
    "can't use", "cannot use", "unable to use", "lost money",
    "money deducted", "business is affected", "missed my", "urgent",
    "very important", "खाते से पैसे", "पैसे कट",
)
AGENT_COMMITMENT_CUES = (
    "we will call", "we'll call", "i will call", "i'll call",
    "we will refund", "we'll refund", "i will refund", "i'll refund",
    "will be resolved", "within 24 hours", "within 48 hours",
    "by tomorrow", "we have raised a ticket", "ticket has been raised",
    "मैं कॉल", "हम कॉल करेंगे",
)


def _sentences(text: str) -> List[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?।])\\s+|\\n+", text or "")
        if part.strip()
    ]


def _contains(text: str, phrase: str) -> bool:
    """Match phrases without treating short words as arbitrary substrings."""
    escaped = re.escape(phrase.casefold())
    if re.fullmatch(r"[a-z0-9 ]+", phrase.casefold()):
        pattern = r"(?<![a-z0-9])" + escaped.replace(r"\\ ", r"\\s+") + r"(?![a-z0-9])"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    return phrase.casefold() in text


def _evidence(sentences: Iterable[str], cues: Iterable[str], limit: int = 3) -> List[Dict[str, object]]:
    found = []
    seen = set()
    for sentence in sentences:
        matches = [cue for cue in cues if _contains(sentence.casefold(), cue)]
        if matches and sentence not in seen:
            found.append({"text": sentence, "matched_terms": matches[:4]})
            seen.add(sentence)
        if len(found) >= limit:
            break
    return found


def _brief(source: str, limit: int = 420) -> str:
    parts = _sentences(source)
    value = " ".join(parts[:2]) if parts else "No usable transcript is available."
    if len(value) > limit:
        value = value[: limit - 3].rsplit(" ", 1)[0] + "..."
    return value


def analyze_conversation_text(transcript, translated_text=None):
    """Return a stable, JSON-serializable support-call review structure."""
    original = (transcript or "").strip()
    english = (translated_text or "").strip()
    # English translation can improve lexical coverage, but original transcript
    # remains the primary evidence and is always retained in the result.
    analysis_source = english or original
    sentences = _sentences(analysis_source)
    lowered = analysis_source.casefold()

    topics = []
    for topic, phrases in TOPIC_KEYWORDS.items():
        matched = [phrase for phrase in phrases if _contains(lowered, phrase)]
        if matched:
            topics.append({
                "topic": topic,
                "matched_terms": list(dict.fromkeys(matched))[:5],
                "evidence": _evidence(sentences, matched, limit=2),
                "method": "keyword rules",
            })

    resolution_evidence = _evidence(sentences, RESOLUTION_CUES)
    issue_evidence = _evidence(sentences, ISSUE_CUES)
    # A resolution phrase suppresses only an unresolved cue when it appears
    # in the same or a later sentence; it does not erase the underlying issue.
    unresolved_evidence = []
    for item in issue_evidence:
        idx = sentences.index(item["text"])
        resolved_later = any(
            _evidence([s], RESOLUTION_CUES, limit=1)
            for s in sentences[idx:]
        )
        if not resolved_later:
            unresolved_evidence.append(item)

    signal_specs = [
        ("Possible unresolved issue", unresolved_evidence,
         "Issue-related wording appears without a later explicit resolution cue."),
        ("Possible repeat contact", _evidence(sentences, REPEAT_CONTACT_CUES),
         "The speaker may be referring to earlier contact attempts."),
        ("Possible follow-up needed", _evidence(sentences, FOLLOW_UP_CUES),
         "The transcript contains a request or wording associated with a next step."),
        ("Possible escalation", _evidence(sentences, ESCALATION_CUES),
         "Escalation-related wording appears and may need supervisor review."),
        ("Customer-reported impact", _evidence(sentences, CUSTOMER_IMPACT_CUES),
         "The speaker describes a possible impact or urgency."),
        ("Agent commitment mentioned", _evidence(sentences, AGENT_COMMITMENT_CUES),
         "The transcript contains wording that may represent an agent commitment; verify owner and deadline."),
    ]
    signals = []
    for label, evidence, description in signal_specs:
        if evidence:
            signals.append({
                "label": label,
                "evidence": [term for item in evidence for term in item["matched_terms"]][:5],
                "evidence_sentences": evidence,
                "description": description,
                "status": "needs_human_review",
            })

    needs_review = bool(signals) or not bool(original) or not bool(english)
    review_reasons = []
    if not original:
        review_reasons.append("No original transcript is available.")
    if not english:
        review_reasons.append("English meaning is unavailable; findings use the original transcript.")
    if signals:
        review_reasons.append("One or more support review cues were detected.")

    return {
        "brief": _brief(analysis_source),
        "brief_source": "English translation" if english else "original transcript",
        "topics": topics,
        "signals": signals,
        "needs_human_review": needs_review,
        "review_reasons": review_reasons,
        "resolution_cues": resolution_evidence,
        "analysis_source": "English translation" if english else "original transcript",
        "method": "deterministic keyword rules and extractive evidence; not a trained classifier",
        "original_transcript": original,
        "translation_available": bool(english),
        "case_review": {
            "suggested_status": (
                "review_required" if needs_review else "no_configured_cue_detected"
            ),
            "suggested_next_step": (
                "Human reviewer: verify the issue, evidence, resolution state and any promised follow-up against the audio."
                if needs_review else
                "No configured review cue was detected; this is not proof that no action is required."
            ),
        },
    }
