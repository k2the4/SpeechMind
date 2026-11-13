import unittest
from models.romanized_hindi import transliterate_romanized_hindi
from models.language_spans import annotate_language_spans


class FakeEngine:
    def translit_word(self, word, lang_code="hi", topk=1):
        return {"mujhe": ["मुझे"], "chahiye": ["चाहिए"]}.get(word.lower(), [word])


class RomanizedHindiTests(unittest.TestCase):
    def test_only_tagged_hindi_words_are_transliterated(self):
        text = "mujhe project chahiye, please"
        spans = annotate_language_spans(text, "en")["language_spans"]
        result = transliterate_romanized_hindi(text, spans, engine=FakeEngine())
        self.assertTrue(result["available"])
        self.assertEqual(result["text"], "मुझे project चाहिए, please")
        self.assertEqual(result["replacements"], 2)

    def test_english_only_text_is_unchanged(self):
        text = "This is a project layout"
        spans = annotate_language_spans(text, "en")["language_spans"]
        result = transliterate_romanized_hindi(text, spans, engine=FakeEngine())
        self.assertFalse(result["available"])
        self.assertEqual(result["text"], text)

    def test_empty_spans_return_unavailable(self):
        result = transliterate_romanized_hindi("hello", [], engine=FakeEngine())
        self.assertFalse(result["available"])
        self.assertTrue(result["error"])


if __name__ == "__main__":
    unittest.main()
