import unittest

from scripts.asr_error_analysis import align_word_errors, build_word_error_profile


class ASRErrorAnalysisTests(unittest.TestCase):
    def test_word_alignment_reports_substitution_deletion_and_insertion(self):
        result = align_word_errors("one two three", "one too")
        self.assertEqual(result["word_errors"], 2)
        self.assertEqual(result["substitutions"][("two", "too")], 1)
        self.assertEqual(result["deletions"]["three"], 1)
        self.assertEqual(sum(result["insertions"].values()), 0)

    def test_insertions_are_reported(self):
        result = align_word_errors("one two", "one extra two")
        self.assertEqual(result["word_errors"], 1)
        self.assertEqual(result["insertions"]["extra"], 1)

    def test_empty_hypothesis_is_all_deletions(self):
        result = align_word_errors("one two", "")
        self.assertEqual(result["word_errors"], 2)
        self.assertEqual(result["deletions"]["one"], 1)
        self.assertEqual(result["deletions"]["two"], 1)

    def test_multi_reference_uses_wer_selected_alternative(self):
        rows = [{
            "reference_transcript": "one two   one too",
            "srota_hypothesis": "one too",
            "srota_status": "ok",
        }]
        result = build_word_error_profile(
            rows, "srota_hypothesis", "srota_status"
        )
        self.assertEqual(result["scored_samples"], 1)
        self.assertEqual(result["substitutions"], [])
        self.assertEqual(result["deletions"], [])

    def test_failed_rows_are_excluded_from_token_profile(self):
        rows = [
            {
                "reference_transcript": "one two",
                "srota_hypothesis": "one too",
                "srota_status": "ok",
            },
            {
                "reference_transcript": "one two",
                "srota_hypothesis": "",
                "srota_status": "error",
            },
        ]
        result = build_word_error_profile(
            rows, "srota_hypothesis", "srota_status"
        )
        self.assertEqual(result["scored_samples"], 1)
        self.assertEqual(result["substitutions"][0]["reference_token"], "two")
        self.assertEqual(result["substitutions"][0]["hypothesis_token"], "too")


if __name__ == "__main__":
    unittest.main()
