import unittest

from models.translation import FLORES_CODE, Translator


class FakeTranslationProvider:
    def __init__(self):
        self.calls = []

    def translate_to_english(self, text, src_lang):
        self.calls.append((text, src_lang))
        return {
            "translated_text": "This is our thought about this image.",
            "inference_seconds": 1.25,
        }


class TranslationAdapterTests(unittest.TestCase):
    def test_hindi_maps_to_flores_code_and_translation_is_delegated(self):
        provider = FakeTranslationProvider()
        translator = Translator(provider=provider)
        original = "इस image के बारे में हमारा ये thought है"

        translated = translator.to_english(original, "hi")

        self.assertEqual(translated, "This is our thought about this image.")
        self.assertEqual(provider.calls, [(original, "hin_Deva")])
        self.assertTrue(translator.loaded)
        self.assertEqual(translator.last_inference_seconds, 1.25)

    def test_english_is_returned_without_translation_worker(self):
        provider = FakeTranslationProvider()
        translator = Translator(provider=provider)
        original = "I don't know where we are going tomorrow."

        self.assertEqual(translator.to_english(original, "en"), original)
        self.assertEqual(provider.calls, [])
        self.assertFalse(translator.loaded)

    def test_unsupported_language_is_not_silently_passed_off_as_english(self):
        translator = Translator(provider=FakeTranslationProvider())
        with self.assertRaises(ValueError):
            translator.to_english("sample text", "xx")

    def test_hindi_flores_code_is_defined(self):
        self.assertEqual(FLORES_CODE["hi"], "hin_Deva")


if __name__ == "__main__":
    unittest.main()
