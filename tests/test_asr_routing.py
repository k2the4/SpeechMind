import unittest

from models.asr_routing import (
    srota_supports_duration,
    whisper_transcript_suggests_hinglish,
)


class AsrRoutingSignalTests(unittest.TestCase):
    def test_clear_english_transcript_does_not_trigger_srota(self):
        self.assertFalse(
            whisper_transcript_suggests_hinglish(
                "This is a useful image for our project", "en"
            )
        )

    def test_romanized_hindi_in_whisper_transcript_triggers_srota(self):
        self.assertTrue(
            whisper_transcript_suggests_hinglish(
                "main ghar ja raha hoon", "en"
            )
        )

    def test_mixed_script_hinglish_in_whisper_transcript_triggers_srota(self):
        self.assertTrue(
            whisper_transcript_suggests_hinglish(
                "मेरा favourite festival Diwali है", "en"
            )
        )

    def test_unknown_latin_tokens_do_not_trigger_srota_alone(self):
        self.assertFalse(
            whisper_transcript_suggests_hinglish(
                "Xyzz blorpt flarm", "en"
            )
        )

    def test_srota_duration_limit_is_inclusive(self):
        self.assertTrue(srota_supports_duration(30 * 16000))
        self.assertFalse(srota_supports_duration(30 * 16000 + 1))
        self.assertFalse(srota_supports_duration(0))
        self.assertFalse(srota_supports_duration(16000, sample_rate=0))


if __name__ == "__main__":
    unittest.main()
