import unittest

from models.text_emotion_routing import resolve_text_emotion_language


class TextEmotionRoutingTests(unittest.TestCase):
    def test_hinglish_with_english_primary_is_not_sent_to_english_fallback(self):
        route = resolve_text_emotion_language(
            "en",
            languages_present=["en", "hi"],
            code_switch_detected=True,
            supported_languages=["en"],
        )
        self.assertFalse(route["available"])
        self.assertEqual(route["language"], "hinglish")
        self.assertIn("withheld", route["reason"])

    def test_hinglish_with_hindi_primary_requires_explicit_hinglish_support(self):
        route = resolve_text_emotion_language(
            "hi",
            languages_present=["en", "hi"],
            code_switch_detected=True,
            supported_languages=["hi"],
        )
        self.assertFalse(route["available"])
        self.assertEqual(route["language"], "hinglish")

    def test_romanized_hindi_requires_explicit_hinglish_support(self):
        route = resolve_text_emotion_language(
            "en",
            languages_present=["hi"],
            romanized_hindi_detected=True,
            supported_languages=["en"],
        )
        self.assertFalse(route["available"])
        self.assertEqual(route["language"], "hinglish")

    def test_hinglish_is_allowed_when_checkpoint_declares_support(self):
        route = resolve_text_emotion_language(
            "en",
            languages_present=["en", "hi"],
            code_switch_detected=True,
            supported_languages=["en", "hi", "hinglish"],
        )
        self.assertTrue(route["available"])
        self.assertEqual(route["language"], "hinglish")

    def test_pure_english_keeps_english_route(self):
        route = resolve_text_emotion_language(
            "en",
            languages_present=["en"],
            supported_languages=["en"],
        )
        self.assertTrue(route["available"])
        self.assertEqual(route["language"], "en")

    def test_pure_hindi_keeps_hindi_route(self):
        route = resolve_text_emotion_language(
            "hi",
            languages_present=["hi"],
            supported_languages=["hi"],
        )
        self.assertTrue(route["available"])
        self.assertEqual(route["language"], "hi")

    def test_other_mixed_language_combination_is_withheld(self):
        route = resolve_text_emotion_language(
            "pa",
            languages_present=["en", "pa"],
            code_switch_detected=True,
            supported_languages=["en", "pa"],
        )
        self.assertFalse(route["available"])
        self.assertEqual(route["language"], "unsupported_mixed")


if __name__ == "__main__":
    unittest.main()
