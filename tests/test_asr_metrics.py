import unittest

from scripts.asr_metrics import (
    aggregate_error_counts,
    edit_distance,
    error_counts,
    normalize_asr_text,
)


class ASRMetricTests(unittest.TestCase):
    def test_unicode_normalization_and_punctuation(self):
        self.assertEqual(normalize_asr_text("नमस्ते, दुनिया।"), "नमस्ते दुनिया")

    def test_edit_distance(self):
        self.assertEqual(edit_distance(["a", "b", "c"], ["a", "x", "c"]), 1)
        self.assertEqual(edit_distance([], ["a"]), 1)

    def test_error_counts(self):
        result = error_counts("मैं कल जाऊँगा।", "मैं आज जाऊँगा।")
        self.assertEqual(result["reference_words"], 3)
        self.assertEqual(result["word_errors"], 1)
        self.assertGreater(result["cer"], 0.0)

    def test_aggregate_uses_corpus_level_denominator(self):
        rows = [
            error_counts("a b c", "a b x"),
            error_counts("a", "x"),
        ]
        result = aggregate_error_counts(rows)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["wer"], 2 / 4)


if __name__ == "__main__":
    unittest.main()
