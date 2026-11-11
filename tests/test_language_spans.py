import unittest

from models.language_spans import annotate_language_spans


class LanguageSpanTests(unittest.TestCase):
    def test_mixed_script_hinglish_has_language_and_script_spans(self):
        text = "मेरा favourite festival Diwali है"
        result = annotate_language_spans(text, "hi")

        self.assertTrue(result["mixed_script_detected"])
        self.assertTrue(result["code_switch_detected"])
        self.assertEqual(result["dominant_language"], "hi")
        self.assertEqual(result["language_evidence"]["hi"], 2)
        self.assertEqual(result["language_evidence"]["en"], 1)
        for span in result["language_spans"]:
            self.assertEqual(
                text[span["char_start"]:span["char_end"]],
                span["text"],
            )

    def test_romanized_hindi_is_detected_without_calling_it_code_switch(self):
        result = annotate_language_spans("main ghar ja raha hoon", "en")

        self.assertTrue(result["romanized_hindi_detected"])
        self.assertEqual(result["dominant_language"], "hi")
        self.assertFalse(result["code_switch_detected"])
        self.assertTrue(
            all(
                span["language"] == "hi"
                for span in result["language_spans"]
            )
        )

    def test_romanized_hindi_mixed_with_english_is_detected(self):
        result = annotate_language_spans(
            "main meeting mein ja raha hoon", "en"
        )

        self.assertTrue(result["romanized_hindi_detected"])
        self.assertTrue(result["code_switch_detected"])
        self.assertIn("hi", result["languages_present"])
        self.assertIn("en", result["languages_present"])

    def test_pure_english_remains_english(self):
        result = annotate_language_spans(
            "This is our thought about this image", "en"
        )

        self.assertEqual(result["languages_present"], ["en"])
        self.assertFalse(result["code_switch_detected"])
        self.assertFalse(result["romanized_hindi_detected"])

    def test_pure_hindi_remains_hindi(self):
        result = annotate_language_spans("यह मेरी किताब है", "hi")

        self.assertEqual(result["languages_present"], ["hi"])
        self.assertFalse(result["code_switch_detected"])
        self.assertFalse(result["mixed_script_detected"])

    def test_ambiguous_latin_words_are_not_forced_to_english(self):
        result = annotate_language_spans("Diwali Xyzz", "hi")

        self.assertEqual(result["languages_present"], [])
        self.assertTrue(
            all(span["language"] == "und" for span in result["language_spans"])
        )

    def test_offsets_are_character_offsets_not_audio_timestamps(self):
        text = "main meeting mein ja raha hoon."
        result = annotate_language_spans(text, "en")

        self.assertEqual(result["span_unit"], "transcript_character_offsets")
        self.assertFalse(result["audio_timestamps_available"])
        self.assertNotIn("start_time", result["language_spans"][0])
        self.assertEqual(result["language_spans"][0]["char_start"], 0)


if __name__ == "__main__":
    unittest.main()
