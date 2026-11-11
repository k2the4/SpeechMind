import unittest

from models.translation_routing import resolve_translation_route


class TranslationRoutingTests(unittest.TestCase):
    def test_english_primary_mixed_script_hindi_english_uses_hindi_source(self):
        route = resolve_translation_route(
            "en",
            language_spans=[
                {"language": "hi", "script": "Devanagari", "text": "मुझे"},
                {"language": "en", "script": "Latin", "text": "the project"},
            ],
        )

        self.assertTrue(route["available"])
        self.assertEqual(route["source_language"], "hi")
        self.assertEqual(route["method"], "indictrans2_mixed_script_best_effort")
        self.assertIn("not yet been evaluated", route["note"])

    def test_romanized_hindi_is_not_sent_directly_to_indictrans2(self):
        route = resolve_translation_route(
            "hi",
            language_spans=[
                {"language": "hi", "script": "Latin", "text": "mujhe"},
                {"language": "en", "script": "Latin", "text": "the project"},
            ],
        )

        self.assertFalse(route["available"])
        self.assertEqual(
            route["method"], "romanized_hindi_transliteration_required"
        )
        self.assertIn("not been sent directly", route["error"])
        self.assertTrue(route["romanized_hindi_detected"])

    def test_romanized_hindi_flag_without_spans_is_respected(self):
        route = resolve_translation_route(
            "en", language_spans=[], romanized_hindi_detected=True
        )

        self.assertFalse(route["available"])
        self.assertEqual(
            route["method"], "romanized_hindi_transliteration_required"
        )

    def test_pure_english_uses_original_transcript_without_worker(self):
        route = resolve_translation_route("en", [])

        self.assertTrue(route["available"])
        self.assertEqual(route["source_language"], "en")
        self.assertEqual(route["method"], "original_english_transcript")

    def test_pure_hindi_uses_indictrans2(self):
        route = resolve_translation_route(
            "hi",
            language_spans=[
                {"language": "hi", "script": "Devanagari", "text": "नमस्ते"},
            ],
        )

        self.assertTrue(route["available"])
        self.assertEqual(route["source_language"], "hi")
        self.assertEqual(route["method"], "indictrans2")
        self.assertIsNone(route["note"])

    def test_non_hindi_indic_language_keeps_its_detected_source_code(self):
        route = resolve_translation_route(
            "bn",
            language_spans=[
                {"language": "bn", "script": "Bengali", "text": "আমি"},
            ],
        )

        self.assertTrue(route["available"])
        self.assertEqual(route["source_language"], "bn")
        self.assertEqual(route["method"], "indictrans2")


if __name__ == "__main__":
    unittest.main()
