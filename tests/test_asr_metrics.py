import unittest

from scripts.asr_metrics import (
    aggregate_error_counts,
    edit_distance,
    error_counts,
    error_counts_multi_reference,
    normalize_asr_text,
    split_reference_alternatives,
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


    def test_hiacc_reference_alternatives_split_on_three_spaces(self):
        alternatives = split_reference_alternatives(
            "चिंटू और राजू।   चिन्टु और राजु।"
        )
        self.assertEqual(
            alternatives,
            ["चिंटू और राजू।", "चिन्टु और राजु।"],
        )

    def test_normal_word_spacing_is_not_split_into_alternatives(self):
        self.assertEqual(
            split_reference_alternatives("this is a normal sentence"),
            ["this is a normal sentence"],
        )

    def test_tab_and_double_pipe_reference_separators_are_supported(self):
        self.assertEqual(
            split_reference_alternatives("पहला वाक्य\tदूसरा वाक्य || तीसरा वाक्य"),
            ["पहला वाक्य", "दूसरा वाक्य", "तीसरा वाक्य"],
        )

    def test_multi_reference_metrics_accept_exact_match_to_one_alternative(self):
        result = error_counts_multi_reference(
            "चिंटू और राजू।   चिन्टु और राजु।",
            "चिंटू और राजू",
        )
        self.assertEqual(result["reference_alternatives"], 2)
        self.assertEqual(result["wer"], 0.0)
        self.assertEqual(result["cer"], 0.0)
        self.assertEqual(result["wer_reference_index"], 1)
        self.assertEqual(result["cer_reference_index"], 1)

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
