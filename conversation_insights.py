"""Evidence-grounded customer-support case extraction.

The speech pipeline remains the source of transcript, translation, language and
optional emotion outputs. This module adds a separate support-intake layer.
Rules are intentionally conservative: all extracted details are attributed to
the transcript and every finding carries original-transcript evidence.
"""
import re
from typing import Dict, Iterable, List


TOPIC_KEYWORDS = {
    "Payments & refunds": (
        "payment", "paid", "refund", "money", "transaction", "billing", "charge",
        "paisa", "paise", "payment failed", "deducted", "double charged",
        "पैसे", "भुगतान", "रिफंड", "पेमेंट", "पैसा कट", "पैसे कट", "कट गए",
        "रुपये कट", "रुपए कट", "बैंक से पैसे", "खाते से पैसे", "पैसे वापस",
        "paise kat", "paise kate", "payment kar", "refund nahi", "paisa kata",
        "bank account", "money deducted", "amount deducted",
    ),
    "Orders & delivery": (
        "order", "delivery", "delivered", "shipment", "courier", "parcel", "ऑर्डर कन्फर्म", "ऑर्डर कन्फर्म नहीं", "ऑर्डर पक्का",
        "dispatch", "tracking", "not delivered", "आर्डर", "ऑर्डर", "डिलीवरी",
        "ऑर्डर नहीं", "samaan", "product nahi mila", "order nahi mila",
        "delivery nahi", "parcel nahi",
    ),
    "Account access": (
        "account access", "account blocked", "login", "log in", "password",
        "otp", "verification", "blocked", "sign in", "लॉगिन", "पासवर्ड",
        "ओटीपी", "खाता", "login nahi", "otp nahi",
    ),
    "Technical support": (
        "error", "issue", "problem", "not working", "crash", "failed",
        "app", "website", "server", "bug", "लोड नहीं", "काम नहीं कर",
        "kaam nahi kar", "open nahi ho", "chal nahi raha", "hang ho",
    ),
    "Cancellation & returns": (
        "cancel", "cancellation", "return", "replace", "replacement",
        "exchange", "वापस", "रिटर्न", "कैंसल", "cancel kar", "return karna",
    ),
    "Service & agent experience": (
        "agent", "executive", "customer care", "rude",
        "helpful", "behaviour", "behavior", "representative", "customer service complaint",
        "customer care wale", "baat nahi",
    ),
}

ISSUE_CUES = (
    "problem", "issue", "complaint", "not working", "not received", "pending", "अभी भी", "अभी तक", "कन्फर्म नहीं", "resolve नहीं",
    "haven't received", "have not received", "didn't receive", "did not receive",
    "still waiting", "not resolved", "not fixed", "not arrived", "failed",
    "delay", "delayed", "late", "refund nahi", "nahi aaya", "nahi mila",
    "abhi tak", "paise wapas", "काम नहीं कर", "नहीं मिला", "नहीं आया",
    "अभी तक", "पैसे कट", "नहीं हुआ", "दिक्कत", "समस्या", "परेशानी",
    "kaam nahi", "nahi ho raha", "nahi hua", "abhi tak nahi",
)
RESOLUTION_CUES = (
    "issue is resolved", "problem is resolved", "resolved now",
    "received the refund", "refund has arrived", "has been resolved",
    "got the refund", "refund received", "delivered now", "working now",
    "fixed now", "समस्या हल", "समाधान हो गया", "पैसे वापस मिल गए",
    "refund mil gaya", "paise aa gaye", "ab kaam kar raha",
)
REPEAT_CONTACT_CUES = (
    "called again", "called three times", "called multiple times", "twice", "two times", "दो बार", "कई बार",
    "contacted before", "already called", "again and again", "multiple times",
    "third time", "pehle bhi", "पहले भी", "kai baar", "कई बार",
    "dobara call", "baar baar", "बार बार",
)
FOLLOW_UP_CUES = (
    "please check", "please resolve", "call me back", "follow up", "follow-up", "resolution", "confirmation", "आज ही", "बताइए", "बता दीजिए", "चेक करके",
    "update me", "let me know", "still waiting", "not resolved", "abhi tak",
    "nahi mila", "कॉल बैक", "जल्दी बताएं", "update dena", "bata dijiyega",
)
ESCALATION_CUES = (
    "escalate", "escalation", "manager", "supervisor", "consumer court",
    "complaint to", "legal action", "social media", "senior team",
    "मैनेजर", "शिकायत", "manager se baat", "senior se baat",
)
CUSTOMER_IMPACT_CUES = (
    "can't use", "cannot use", "unable to use", "lost money", "money deducted",
    "business is affected", "missed my", "urgent", "very important",
    "खाते से पैसे", "पैसे कट", "बैंक अकाउंट", "जरूरी", "बहुत परेशानी", "काम रुक",
    "kaam ruk", "paise kat",
)
AGENT_COMMITMENT_CUES = (
    "we will call", "we'll call", "i will call", "i'll call",
    "we will refund", "we'll refund", "i will refund", "i'll refund",
    "will be resolved", "within 24 hours", "within 48 hours", "by tomorrow",
    "we have raised a ticket", "ticket has been raised", "मैं कॉल",
    "हम कॉल करेंगे", "ticket raise", "call karenge",
)

TOPIC_QUESTIONS = {
    "Payments & refunds": [
        "What was the payment date and approximate time?",
        "Can you provide the transaction or reference ID?",
        "Which payment method or bank was used?",
    ],
    "Orders & delivery": [
        "Can you share the order ID?",
        "What delivery date was promised?",
        "What does the latest tracking update show?",
    ],
    "Account access": [
        "Which phone number or email is registered on the account?",
        "What happens when you try to sign in?",
        "Do you receive the OTP or verification message?",
    ],
    "Technical support": [
        "Which app, website or device is affected?",
        "When did the problem start?",
        "What exact error message do you see?",
    ],
    "Cancellation & returns": [
        "Can you share the order ID?",
        "When was the cancellation or return requested?",
        "Was a pickup or confirmation message received?",
    ],
    "Service & agent experience": [
        "When did the interaction take place?",
        "Was a previous ticket or reference number provided?",
    ],
}


def _sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?।])\s+|\n+", text or "") if part.strip()]


def _contains(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.casefold())
    if re.fullmatch(r"[a-z0-9 ]+", phrase.casefold()):
        pattern = r"(?<![a-z0-9])" + escaped.replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    return phrase.casefold() in text


def _evidence(sentences: Iterable[str], cues: Iterable[str], limit: int = 3) -> List[Dict[str, object]]:
    found = []
    for sentence in sentences:
        matches = [cue for cue in cues if _contains(sentence.casefold(), cue)]
        if matches:
            found.append({"text": sentence, "matched_terms": matches[:5]})
        if len(found) >= limit:
            break
    return found


def _brief(source: str, limit: int = 420) -> str:
    parts = _sentences(source)
    value = " ".join(parts[:2]) if parts else "No usable transcript is available."
    if len(value) > limit:
        value = value[:limit - 3].rsplit(" ", 1)[0] + "..."
    return value


def _topic_matches(original: str, english: str):
    original_sentences = _sentences(original)
    english_sentences = _sentences(english)
    topics = []
    for topic, phrases in TOPIC_KEYWORDS.items():
        original_matches = [p for p in phrases if _contains(original.casefold(), p)]
        english_matches = [p for p in phrases if _contains(english.casefold(), p)] if english else []
        matches = list(dict.fromkeys(original_matches + english_matches))
        if not matches:
            continue
        # Prefer original-language evidence. Translation-only matches are shown
        # as translated evidence and are never presented as verbatim transcript.
        evidence = _evidence(original_sentences, original_matches, limit=2)
        if not evidence and english_matches:
            evidence = _evidence(english_sentences, english_matches, limit=2)
            evidence_source = "english_translation"
        else:
            evidence_source = "original_transcript"
        topics.append({
            "topic": topic,
            "matched_terms": matches[:6],
            "evidence": evidence,
            "evidence_source": evidence_source,
            "method": "multilingual keyword rules",
        })
    return topics


def analyze_conversation_text(transcript, translated_text=None):
    """Return an additive, JSON-serializable executive case-review structure."""
    original = (transcript or "").strip()
    english = (translated_text or "").strip()
    original_sentences = _sentences(original)
    analysis_source = english or original
    topics = _topic_matches(original, english)

    # Signals are searched in both sources, but evidence prefers the original.
    def combined_evidence(cues, limit=3):
        found = _evidence(original_sentences, cues, limit=limit)
        if found or not english:
            return found
        return _evidence(_sentences(english), cues, limit=limit)

    original_resolution = _evidence(original_sentences, RESOLUTION_CUES)
    original_issue = _evidence(original_sentences, ISSUE_CUES)
    english_sentences = _sentences(english)
    english_resolution = _evidence(english_sentences, RESOLUTION_CUES) if english else []
    english_issue = _evidence(english_sentences, ISSUE_CUES) if english else []
    resolution_evidence = original_resolution or english_resolution
    issue_evidence = original_issue or english_issue
    # Resolution suppresses an issue only when both are found in the same
    # source, where sentence order is meaningful. Cross-language alignment is
    # unavailable, so a translated resolution never silently clears a cue.
    if original_issue:
        unresolved_evidence = [
            item for item in original_issue
            if not any(
                original_sentences.index(resolution["text"]) >= original_sentences.index(item["text"])
                for resolution in original_resolution
            )
        ]
    elif english_issue:
        unresolved_evidence = [
            item for item in english_issue
            if not any(
                english_sentences.index(resolution["text"]) >= english_sentences.index(item["text"])
                for resolution in english_resolution
            )
        ]
    else:
        unresolved_evidence = []
    signal_specs = [
        ("Possible unresolved issue", unresolved_evidence,
         "Issue-related wording appears without a later explicit resolution cue."),
        ("Possible repeat contact", combined_evidence(REPEAT_CONTACT_CUES),
         "The speaker may be referring to earlier contact attempts."),
        ("Possible follow-up needed", combined_evidence(FOLLOW_UP_CUES),
         "The transcript contains a request or wording associated with a next step."),
        ("Possible escalation", combined_evidence(ESCALATION_CUES),
         "Escalation-related wording appears and may need supervisor review."),
        ("Customer-reported impact", combined_evidence(CUSTOMER_IMPACT_CUES),
         "The speaker describes a possible impact or urgency."),
        ("Agent commitment mentioned", combined_evidence(AGENT_COMMITMENT_CUES),
         "The transcript may contain an agent commitment; verify its owner and deadline."),
    ]
    signals = []
    for label, evidence, description in signal_specs:
        if evidence:
            signals.append({
                "label": label,
                "evidence": [term for item in evidence for term in item["matched_terms"]][:6],
                "evidence_sentences": evidence,
                "description": description,
                "status": "needs_human_review",
            })

    matched_topics = [item["topic"] for item in topics]
    questions = []
    for topic in matched_topics:
        for question in TOPIC_QUESTIONS.get(topic, []):
            if question not in questions:
                questions.append(question)
    if not questions:
        questions = [
            "Is this a customer-support issue, a general enquiry, or another type of conversation?",
            "What outcome or action is the speaker asking for, if any?",
            "Is there any relevant reference, date, amount or account detail to verify?",
        ]

    # Only claim details explicitly present in the transcript. Never invent IDs,
    # amounts, dates or account identifiers from a translation.
    detail_patterns = {
        "transaction_id": r"\b(?:transaction|txn|utr|reference)\s*(?:id|number|no\.?)?\s*[:#-]?\s*([A-Z0-9-]{6,})\b",
        "order_id": r"\b(?:order)\s*(?:id|number|no\.?)?\s*[:#-]?\s*([A-Z0-9-]{5,})\b",
        "amount": r"(?:₹|rs\.?\s*)(\d[\d,]*(?:\.\d{1,2})?)|(?<!\d)(\d[\d,]*(?:\.\d{1,2})?)\s*(?:रुपये|रुपए|रूपये|रूपए|रु\.?)(?!\w)",
    }
    extracted_details = {}
    for key, pattern in detail_patterns.items():
        match = re.search(pattern, original, flags=re.IGNORECASE)
        if match:
            value = next((group for group in match.groups() if group), None)
            if value:
                extracted_details[key] = {"value": value, "source": "original_transcript"}

    # An unclassified recording still needs a human decision; absence of a
    # keyword match must never be interpreted as a clean or resolved case.
    needs_review = bool(signals) or bool(topics) or not bool(original) or not bool(topics)
    review_reasons = []
    if not original:
        review_reasons.append("No original transcript is available.")
    if not english:
        review_reasons.append("English meaning is unavailable; topic rules also inspect the original transcript.")
    if signals:
        review_reasons.append("One or more support review cues were detected.")
    if not topics and original:
        review_reasons.append("No configured topic matched; classify the conversation before closing the review.")
    if topics:
        review_reasons.append("One or more possible support topics were detected.")

    primary_topic = topics[0]["topic"] if topics else "Needs classification"
    status = "review_required" if needs_review else "no_configured_cue_detected"
    return {
        "brief": _brief(analysis_source),
        "brief_source": "English translation" if english else "original transcript",
        "topics": topics,
        "signals": signals,
        "needs_human_review": needs_review,
        "review_reasons": review_reasons,
        "resolution_cues": resolution_evidence,
        "analysis_source": "English translation and original transcript" if english else "original transcript",
        "method": "multilingual deterministic keyword rules and extractive evidence; not a trained classifier",
        "original_transcript": original,
        "translation_available": bool(english),
        "case_brief": {
            "primary_category": primary_topic,
            "issue_summary": _brief(analysis_source),
            "reported_details": extracted_details,
            "missing_information": questions[:5],
            "evidence": [
                {"topic": topic["topic"], "source": topic["evidence_source"], "items": topic["evidence"]}
                for topic in topics
            ],
            "review_caveat": "All extracted details are transcript-derived and require executive verification.",
        },
        "case_review": {
            "suggested_status": status,
            "suggested_next_step": (
                ("Classify the conversation first. If it is a support case, verify the reported issue and evidence, then collect only the relevant missing details."
                 if not topics else
                 "Verify the reported issue and evidence, collect missing details, then check the relevant account/order/payment system.")
                if needs_review else
                "No configured review cue was detected; this is not proof that no action is required."
            ),
            "suggested_questions": questions[:5],
        },
    }


def analyze_conversation(transcript, translated_text=None, language="unknown"):
    """Prefer free local LLM analysis; retain deterministic fallback if unavailable."""
    try:
        from local_llm import analyze_with_local_llm
        result = analyze_with_local_llm(transcript, language=language)
        result["llm"]["fallback_used"] = False
        return result
    except Exception as error:
        fallback = analyze_conversation_text(transcript, translated_text)
        fallback["llm"] = {
            "provider": "deterministic_fallback",
            "available": False,
            "fallback_used": True,
            "error": f"{type(error).__name__}: {error}",
            "setup_hint": "Set GEMINI_API_KEY for the Gemini free tier, or install Ollama for optional offline analysis. See docs/LOCAL_LLM_SETUP.md.",
        }
        return fallback
