import unittest

from models.indic_language_id import IndicLanguageIdentifier


class IndicLanguageIdentifierTests(unittest.TestCase):
    def test_native_hindi_plus_english_is_hinglish(self):
        self.assertEqual(
            IndicLanguageIdentifier.summarize_labels(
                ["hin_Deva", "eng_Latn"]
            ),
            "hinglish",
        )

    def test_romanized_hindi_is_treated_as_hinglish_profile(self):
        self.assertEqual(
            IndicLanguageIdentifier.summarize_labels(["hin_Latn"]),
            "hinglish",
        )

    def test_marathi_is_not_mislabeled_as_hindi(self):
        self.assertEqual(
            IndicLanguageIdentifier.summarize_labels(["mar_Deva"]),
            "mr",
        )

    def test_regional_english_code_switch_is_not_called_hinglish(self):
        self.assertEqual(
            IndicLanguageIdentifier.summarize_labels(
                ["tam_Tamil", "eng_Latn"]
            ),
            "code_mixed",
        )


if __name__ == "__main__":
    unittest.main()
