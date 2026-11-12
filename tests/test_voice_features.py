import unittest

from models.voice_features import rank_top_two_scores


class VoiceScoreDiagnosticsTests(unittest.TestCase):
    def test_returns_ordered_top_two_and_margin(self):
        top_two, margin = rank_top_two_scores({
            "fear": 0.2600,
            "disgust": 0.2567,
            "sadness": 0.179,
        })
        self.assertEqual(top_two[0]["label"], "fear")
        self.assertEqual(top_two[1]["label"], "disgust")
        self.assertEqual(margin, 0.0033)

    def test_single_class_has_unit_margin(self):
        top_two, margin = rank_top_two_scores({"neutral": 1.0})
        self.assertEqual(top_two, [{"label": "neutral", "score": 1.0}])
        self.assertEqual(margin, 1.0)

    def test_empty_or_invalid_scores_return_no_ranking(self):
        self.assertEqual(rank_top_two_scores({}), ([], None))
        self.assertEqual(rank_top_two_scores({"fear": "bad"}), ([], None))


if __name__ == "__main__":
    unittest.main()
