import unittest

from conversation_insights import analyze_conversation_text


class ConversationInsightsTests(unittest.TestCase):
    def test_extracts_payment_and_refund_topic(self):
        result = analyze_conversation_text(
            "I made the payment yesterday, but my refund has not arrived."
        )
        topics = [item["topic"] for item in result["topics"]]
        self.assertIn("Payments & refunds", topics)
        self.assertTrue(result["signals"])

    def test_detects_repeat_contact_as_review_cue(self):
        result = analyze_conversation_text(
            "I have called three times already and I am still waiting."
        )
        labels = [item["label"] for item in result["signals"]]
        self.assertIn("Possible repeat contact", labels)
        self.assertIn("Possible follow-up needed", labels)

    def test_uses_translation_when_available_for_brief(self):
        result = analyze_conversation_text(
            "Maine paise diye.", "I paid the money. The refund is still pending."
        )
        self.assertEqual(result["brief_source"], "English translation")
        self.assertIn("I paid the money.", result["brief"])

    def test_empty_transcript_is_flagged_for_review(self):
        result = analyze_conversation_text("")
        self.assertTrue(result["needs_human_review"])
        self.assertEqual(result["topics"], [])

    def test_does_not_claim_emotion_or_verified_intent(self):
        result = analyze_conversation_text("The order arrived yesterday.")
        self.assertNotIn("emotion", result)
        self.assertNotIn("intent", result)


if __name__ == "__main__":
    unittest.main()
