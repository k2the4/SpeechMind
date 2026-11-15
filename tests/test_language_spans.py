import unittest

from models.language_spans import annotate_language_spans


class LanguageSpanTests(unittest.TestCase):
    def test_mixed_script_hinglish_has_language_and_script_spans(self):
        text = "मेरा favourite festival Diwali है"
        result = annotate_language_spans(text, "hi")

        self.assertTrue(result["mixed_script_detected"])
        self.assertTrue(result["code_switch_detected"])
        self.assertEqual(result["dominant_language"], "hi")
        self.assertGreaterEqual(result["language_evidence"]["hi"], 2)
        self.assertGreaterEqual(result["language_evidence"]["en"], 1)
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

    def test_unknown_latin_words_do_not_create_false_english_switch(self):
        result = annotate_language_spans("main blorpt ja raha hoon", "en")

        self.assertTrue(result["romanized_hindi_detected"])
        self.assertFalse(result["code_switch_detected"])
        unknown = [
            span for span in result["language_spans"]
            if "blorpt" in span["tokens"]
        ]
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]["language"], "und")

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

    def test_ambiguous_english_words_the_and_main_are_not_hindi_evidence(self):
        result = annotate_language_spans(
            "This is the main image for our project", "en"
        )

        self.assertEqual(result["dominant_language"], "en")
        self.assertFalse(result["romanized_hindi_detected"])
        self.assertFalse(result["code_switch_detected"])
        self.assertTrue(
            all(
                span["language"] == "en"
                for span in result["language_spans"]
            )
        )

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

    def test_devanagari_vowel_marks_are_preserved_in_spans(self):
        text = "इस image के बारे में हमारे ये thought है"
        result = annotate_language_spans(text, "hi")

        devanagari_spans = [
            span for span in result["language_spans"]
            if span["script"] == "Devanagari"
        ]
        hindi_tokens = [
            token
            for span in devanagari_spans
            for token in span["tokens"]
        ]

        self.assertIn("के", hindi_tokens)
        self.assertIn("में", hindi_tokens)
        self.assertIn("ये", hindi_tokens)
        self.assertIn("है", hindi_tokens)
        self.assertNotIn("य", hindi_tokens)
        self.assertNotIn("ह", hindi_tokens)
        for span in devanagari_spans:
            self.assertEqual(
                text[span["char_start"]:span["char_end"]],
                span["text"],
            )

    def test_offsets_are_character_offsets_not_audio_timestamps(self):
        text = "main meeting mein ja raha hoon."
        result = annotate_language_spans(text, "en")

        self.assertEqual(result["span_unit"], "transcript_character_offsets")
        self.assertFalse(result["audio_timestamps_available"])
        self.assertNotIn("start_time", result["language_spans"][0])
        self.assertEqual(result["language_spans"][0]["char_start"], 0)


    def test_common_design_vocabulary_is_classified_as_english(self):
        text = "I am not sure about the layout and design"
        result = annotate_language_spans(text, "en")
        english_tokens = [
            token
            for span in result["language_spans"]
            for token in span["tokens"]
            if span["language"] == "en"
        ]
        self.assertIn("sure", english_tokens)
        self.assertIn("layout", english_tokens)
        self.assertIn("design", english_tokens)
        self.assertFalse(result["romanized_hindi_detected"])

    def test_wireframe_and_prototype_terms_are_classified_as_english(self):
        result = annotate_language_spans(
            "I am not sure about the wireframe prototype", "en"
        )
        english_tokens = [
            token
            for span in result["language_spans"]
            for token in span["tokens"]
            if span["language"] == "en"
        ]
        self.assertIn("wireframe", english_tokens)
        self.assertIn("prototype", english_tokens)

    def test_common_hinglish_product_terms_are_classified_as_english(self):
        text = (
            "team college dashboard users feedback frames navigation simple "
            "important features test final improve"
        )
        result = annotate_language_spans(text, "hi")
        english_tokens = {
            token
            for span in result["language_spans"]
            for token in span["tokens"]
            if span["language"] == "en"
        }
        for term in (
            "team", "college", "dashboard", "users", "feedback", "frames",
            "navigation", "simple", "important", "features", "test", "final",
            "improve",
        ):
            with self.subTest(term=term):
                self.assertIn(term, english_tokens)



    def test_devanagari_transliterated_english_detects_code_switch(self):
        text = (
            "मैंने कस्टमर सपोर्ट से कॉन्टेक्ट किया "
            "लेकिन मेरा इशू रिज़ॉल्व नहीं हुआ"
        )
        result = annotate_language_spans(text, "hi")

        self.assertTrue(result["code_switch_detected"])
        self.assertFalse(result["mixed_script_detected"])
        self.assertIn("hi", result["languages_present"])
        self.assertIn("en", result["languages_present"])

        english_tokens = [
            token
            for span in result["language_spans"]
            if span["language"] == "en"
            for token in span["tokens"]
        ]
        self.assertIn("कस्टमर", english_tokens)
        self.assertIn("सपोर्ट", english_tokens)
        self.assertIn("कॉन्टेक्ट", english_tokens)
        self.assertIn("इशू", english_tokens)

    def test_devanagari_respects_marathi_and_nepali_language_hints(self):
        marathi = annotate_language_spans("माझे घर सुंदर आहे", "mr")
        nepali = annotate_language_spans("मेरो घर राम्रो छ", "ne")

        self.assertEqual(marathi["languages_present"], ["mr"])
        self.assertEqual(nepali["languages_present"], ["ne"])
        self.assertTrue(all(s["script"] == "Devanagari" for s in marathi["language_spans"]))
        self.assertTrue(all(s["script"] == "Devanagari" for s in nepali["language_spans"]))

    def test_shared_bengali_script_respects_assamese_hint(self):
        assamese = annotate_language_spans("মই আজি ঘৰলৈ যাওঁ", "as")
        bengali = annotate_language_spans("আমি আজ বাড়ি যাই", "bn")

        self.assertEqual(assamese["languages_present"], ["as"])
        self.assertEqual(bengali["languages_present"], ["bn"])
        self.assertTrue(all(s["script"] == "Bengali" for s in assamese["language_spans"]))

    def test_odia_script_is_recognized(self):
        result = annotate_language_spans("ମୁଁ ଘରକୁ ଯାଉଛି", "or")

        self.assertEqual(result["languages_present"], ["or"])
        self.assertTrue(all(s["script"] == "Odia" for s in result["language_spans"]))

    def test_offsets_remain_exact_with_emoji_and_unicode_punctuation(self):
        text = "🙂 नमस्ते—please, मेरा design!"
        result = annotate_language_spans(text, "hi")

        for span in result["language_spans"]:
            self.assertEqual(text[span["char_start"]:span["char_end"]], span["text"])
            self.assertGreater(span["char_end"], span["char_start"])

    def test_transliterated_english_preserves_devanagari_script(self):
        result = annotate_language_spans("मेरा पेमेंट पेंडिंग है", "hi")
        english_spans = [
            span for span in result["language_spans"]
            if span["language"] == "en"
        ]

        self.assertTrue(english_spans)
        self.assertTrue(
            all(span["script"] == "Devanagari" for span in english_spans)
        )


if __name__ == "__main__":
    unittest.main()
