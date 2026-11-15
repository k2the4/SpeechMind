import unittest

from conversation_insights import analyze_conversation_text


class ConversationInsightsTests(unittest.TestCase):
    def test_extracts_payment_and_refund_topic_with_evidence(self):
        result = analyze_conversation_text(
            "I made the payment yesterday. My refund has not arrived."
        )
        topic = next(x for x in result["topics"] if x["topic"] == "Payments & refunds")
        self.assertTrue(topic["evidence"])
        self.assertIn("Possible unresolved issue", [x["label"] for x in result["signals"]])

    def test_detects_repeat_contact_and_follow_up(self):
        result = analyze_conversation_text(
            "I have called three times already. I am still waiting."
        )
        labels = [item["label"] for item in result["signals"]]
        self.assertIn("Possible repeat contact", labels)
        self.assertIn("Possible follow-up needed", labels)

    def test_detects_hinglish_payment_and_refund_cues(self):
        result = analyze_conversation_text(
            "Maine payment kar di but refund abhi tak nahi aaya."
        )
        topics = [item["topic"] for item in result["topics"]]
        self.assertIn("Payments & refunds", topics)
        self.assertTrue(result["signals"])

    def test_detects_escalation_and_customer_impact(self):
        result = analyze_conversation_text(
            "My money was deducted and I cannot use the service. Please escalate this to your manager."
        )
        labels = [item["label"] for item in result["signals"]]
        self.assertIn("Possible escalation", labels)
        self.assertIn("Customer-reported impact", labels)

    def test_detects_agent_commitment(self):
        result = analyze_conversation_text(
            "We will call you back within 24 hours."
        )
        labels = [item["label"] for item in result["signals"]]
        self.assertIn("Agent commitment mentioned", labels)

    def test_resolution_phrase_prevents_unresolved_status(self):
        result = analyze_conversation_text(
            "My refund was delayed. I received the refund today."
        )
        labels = [item["label"] for item in result["signals"]]
        self.assertNotIn("Possible unresolved issue", labels)
        self.assertTrue(result["resolution_cues"])

    def test_issue_without_resolution_remains_review_cue(self):
        result = analyze_conversation_text(
            "The payment failed yesterday and I am still waiting."
        )
        labels = [item["label"] for item in result["signals"]]
        self.assertIn("Possible unresolved issue", labels)

    def test_does_not_match_short_topic_words_inside_other_words(self):
        result = analyze_conversation_text("I am happy with the support.")
        topics = [item["topic"] for item in result["topics"]]
        self.assertNotIn("Technical support", topics)

    def test_uses_translation_for_analysis_but_preserves_original(self):
        result = analyze_conversation_text(
            "Maine paise diye.",
            "I paid the money. The refund is still pending.",
        )
        self.assertEqual(result["brief_source"], "English translation")
        self.assertEqual(result["original_transcript"], "Maine paise diye.")
        self.assertIn("Payments & refunds", [x["topic"] for x in result["topics"]])

    def test_empty_transcript_is_flagged_for_review(self):
        result = analyze_conversation_text("")
        self.assertTrue(result["needs_human_review"])
        self.assertEqual(result["topics"], [])
        self.assertEqual(result["case_review"]["suggested_status"], "review_required")

    def test_no_cues_is_not_claimed_as_verified_resolution(self):
        result = analyze_conversation_text(
            "Hello, thank you for contacting customer support."
        )
        self.assertFalse(result["signals"])
        self.assertEqual(
            result["case_review"]["suggested_status"],
            "no_configured_cue_detected",
        )

    def test_does_not_claim_emotion_or_verified_intent(self):
        result = analyze_conversation_text("The order arrived yesterday.")
        self.assertNotIn("emotion", result)
        self.assertNotIn("intent", result)


if __name__ == "__main__":
    unittest.main()
