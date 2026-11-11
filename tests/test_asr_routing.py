import unittest
from pathlib import Path

from models.asr_routing import (
    audio_language_suggests_hindi,
    srota_supports_duration,
    srota_transcript_supports_hindi,
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

    def test_ambiguous_english_word_main_does_not_trigger_srota(self):
        self.assertFalse(
            whisper_transcript_suggests_hinglish(
                "This is the main image for our project", "en"
            )
        )

    def test_mixed_script_with_unknown_english_name_triggers_srota(self):
        self.assertTrue(
            whisper_transcript_suggests_hinglish("मेरा iPhone है", "hi")
        )


    def test_short_english_detection_with_secondary_hindi_audio_score_triggers_probe(self):
        self.assertTrue(
            audio_language_suggests_hindi(
                {"en": 0.78, "hi": 0.18}, "en", 1.4
            )
        )

    def test_weak_hindi_audio_score_does_not_trigger_probe(self):
        self.assertFalse(
            audio_language_suggests_hindi(
                {"en": 0.91, "hi": 0.04}, "en", 1.4
            )
        )

    def test_audio_level_hindi_probe_is_limited_to_short_clips(self):
        self.assertFalse(
            audio_language_suggests_hindi(
                {"en": 0.78, "hi": 0.18}, "en", 6.1
            )
        )

    def test_audio_level_hindi_probe_only_applies_to_english_primary(self):
        self.assertFalse(
            audio_language_suggests_hindi(
                {"hi": 0.78, "en": 0.18}, "hi", 1.4
            )
        )

    def test_srota_devanagari_only_output_is_accepted_as_hindi(self):
        self.assertTrue(
            srota_transcript_supports_hindi("चिंटू और राजू।", "en")
        )

    def test_srota_english_only_output_is_not_accepted_as_hindi(self):
        self.assertFalse(
            srota_transcript_supports_hindi(
                "That is why the floor looks wet", "en"
            )
        )

    def test_srota_duration_limit_is_inclusive(self):
        self.assertTrue(srota_supports_duration(30 * 16000))
        self.assertFalse(srota_supports_duration(30 * 16000 + 1))
        self.assertFalse(srota_supports_duration(0))
        self.assertFalse(srota_supports_duration(16000, sample_rate=0))


class AsrRouterIntegrationPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.source = (root / "models" / "asr.py").read_text(encoding="utf-8")

    def test_confident_english_decodes_before_transcript_triggered_probe(self):
        branch_start = self.source.index('if detected_language in ("en", "english"):')
        branch_end = self.source.index(
            'if indic_language in INDIC_LANGUAGES and language_score >= MIN_INDIC_ROUTE_SCORE:',
            branch_start,
        )
        branch = self.source[branch_start:branch_end]
        self.assertLess(
            branch.index('self._decode_with_whisper('),
            branch.index('whisper_transcript_suggests_hinglish('),
        )
        self.assertIn('probe_source="whisper_transcript"', branch)

    def test_long_clip_is_skipped_before_calling_srota_provider(self):
        method_start = self.source.index('def _try_srota_code_switch(')
        method_end = self.source.index('def _decode_with_whisper(', method_start)
        method = self.source[method_start:method_end]
        self.assertLess(
            method.index('srota_supports_duration('),
            method.index('self.srota_provider.transcribe_audio('),
        )

    def test_transient_srota_error_does_not_disable_provider_for_session(self):
        method_start = self.source.index('def _try_srota_code_switch(')
        method_end = self.source.index('def _decode_with_whisper(', method_start)
        method = self.source[method_start:method_end]
        handler = method[method.index('except Exception as error:'):]
        self.assertNotIn('self.srota_provider_error =', handler)




if __name__ == "__main__":
    unittest.main()
