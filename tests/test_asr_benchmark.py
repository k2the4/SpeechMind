import unittest

from scripts.benchmark_asr import normalize_text, score_pair, aggregate


class ASRBenchmarkTests(unittest.TestCase):
    def test_normalization_preserves_non_latin_letters(self):
        self.assertEqual(
            normalize_text("कल Exam!"),
            "कल exam",
        )

    def test_wer_and_cer_for_identical_text(self):
        result = score_pair("kal mera exam hai", "kal mera exam hai")
        self.assertEqual(result["wer"], 0.0)
        self.assertEqual(result["cer"], 0.0)

    def test_wer_counts_word_substitutions(self):
        result = score_pair(
            "tomorrow is my exam",
            "gamora is my exam",
        )
        self.assertAlmostEqual(result["wer"], 0.25)

    def test_aggregate_uses_corpus_error_counts(self):
        rows = [
            {
                "word_errors": 1, "reference_words": 4,
                "character_errors": 2, "reference_characters": 10,
                "latency_sec": 2.0,
            },
            {
                "word_errors": 2, "reference_words": 6,
                "character_errors": 3, "reference_characters": 15,
                "latency_sec": 4.0,
            },
        ]
        summary = aggregate(rows)
        self.assertAlmostEqual(summary["wer"], 0.30)
        self.assertAlmostEqual(summary["cer"], 5 / 25)
        self.assertAlmostEqual(summary["mean_latency_sec"], 3.0)


if __name__ == "__main__":
    unittest.main()
