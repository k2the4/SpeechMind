import unittest
import numpy as np

from models.bodhan_client import BodhanClient


class BodhanClientTests(unittest.TestCase):
    def test_mixed_devanagari_and_latin_is_not_mislabeled_as_english(self):
        result = BodhanClient.language_profile(
            "Kal mera exam hai and I haven't prepared anything"
        )
        self.assertEqual(result["language_profile"], "english_or_romanized_hinglish")

        result = BodhanClient.language_profile(
            "कल मेरा exam है and I haven't prepared anything"
        )
        self.assertEqual(result["language_profile"], "code_mixed_or_mixed_script")

    def test_native_scripts_are_detected_as_script_evidence(self):
        cases = [
            ("আমার পরীক্ষা কাল", "bn"),
            ("எனக்கு நாளை தேர்வு", "ta"),
            ("నాకు రేపు పరీక్ష", "te"),
            ("ਮੈਨੂੰ ਕੱਲ੍ਹ ਇਮਤਿਹਾਨ ਹੈ", "pa"),
        ]
        for text, expected in cases:
            with self.subTest(expected=expected):
                result = BodhanClient.language_profile(text)
                self.assertEqual(result["language_profile"], expected)

    def test_overlapping_chunk_text_is_deduplicated(self):
        merged = BodhanClient._merge_chunk_text(
            "I was going to college because",
            "to college because it started raining",
        )
        self.assertEqual(
            merged,
            "I was going to college because it started raining",
        )

    def test_long_audio_is_split_below_api_limit_with_overlap(self):
        audio = np.zeros(65 * 16000, dtype=np.float32)
        chunks = BodhanClient._audio_chunks(audio)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk) <= 27 * 16000 for chunk in chunks))
        self.assertGreater(len(chunks[1]), 0)


if __name__ == "__main__":
    unittest.main()
