import unittest

from models.hinglish_context import (
    build_hinglish_context,
    extract_hinglish_context_terms,
    infer_hinglish_context_from_audio_tail,
)


class HinglishContextTests(unittest.TestCase):
    def test_extracts_exact_allowlisted_terms_in_spoken_order(self):
        transcript = "not sure about the layout, then open Figma and the website"
        self.assertEqual(
            extract_hinglish_context_terms(transcript),
            ["layout", "Figma", "website"],
        )

    def test_normalizes_common_compound_spellings(self):
        self.assertEqual(
            build_hinglish_context("Use the wire frame and front end"),
            "wireframe, frontend",
        )

    def test_does_not_match_terms_inside_larger_words(self):
        self.assertEqual(
            build_hinglish_context("The applet has a layoutish issue"),
            "",
        )

    def test_does_not_invent_context_when_no_allowlisted_term_is_present(self):
        self.assertEqual(
            build_hinglish_context("मुझे समझ नहीं आ रहा है"),
            "",
        )

    def test_limits_context_to_a_small_number_of_terms(self):
        self.assertEqual(
            extract_hinglish_context_terms(
                "project design layout interface wireframe prototype Figma",
                max_terms=3,
            ),
            ["project", "design", "layout"],
        )

    def test_whisper_tail_probe_extracts_only_allowlisted_terms(self):
        class FakeWhisper:
            def __init__(self):
                self.kwargs = None
                self.sample_count = None

            def transcribe(self, audio, **kwargs):
                self.sample_count = len(audio)
                self.kwargs = kwargs
                return {"text": "not sure about the layout", "language": "en"}

        audio = [0.0] * (9 * 16000)
        model = FakeWhisper()
        result = infer_hinglish_context_from_audio_tail(
            audio, model,
            language_detector=lambda tail: ("en", 0.94, {"en": 0.94}),
        )

        self.assertEqual(result["context"], "layout")
        self.assertEqual(result["source"], "whisper_tail_vocabulary")
        self.assertEqual(result["transcript"], "not sure about the layout")
        self.assertEqual(model.sample_count, 3 * 16000)
        self.assertEqual(model.kwargs["language"], "en")
        self.assertEqual(result["detected_language"], "en")
        self.assertEqual(result["language_score"], 0.94)
        self.assertEqual(model.kwargs["beam_size"], 1)

    def test_whisper_tail_probe_does_not_add_context_for_generic_english(self):
        class FakeWhisper:
            def transcribe(self, audio, **kwargs):
                return {"text": "I am not sure what to do next", "language": "en"}

        result = infer_hinglish_context_from_audio_tail(
            [0.0] * (8 * 16000),
            FakeWhisper(),
            language_detector=lambda tail: ("en", 0.91, {"en": 0.91}),
        )
        self.assertEqual(result["context"], "")
        self.assertEqual(result["source"], "whisper_tail_no_allowlisted_terms")

    def test_hindi_tail_skips_autoregressive_whisper_decode(self):
        class FakeWhisper:
            def __init__(self):
                self.transcribe_called = False

            def transcribe(self, audio, **kwargs):
                self.transcribe_called = True
                raise AssertionError("Hindi tails should not be decoded for context")

        model = FakeWhisper()
        result = infer_hinglish_context_from_audio_tail(
            [0.0] * (8 * 16000),
            model,
            language_detector=lambda tail: ("hi", 0.97, {"hi": 0.97}),
        )
        self.assertEqual(result["context"], "")
        self.assertEqual(result["source"], "whisper_tail_not_english")
        self.assertEqual(result["detected_language"], "hi")
        self.assertFalse(model.transcribe_called)

    def test_uncertain_english_tail_skips_vocabulary_decode(self):
        class FakeWhisper:
            def transcribe(self, audio, **kwargs):
                raise AssertionError("Uncertain tails should not be decoded")

        result = infer_hinglish_context_from_audio_tail(
            [0.0] * (8 * 16000),
            FakeWhisper(),
            language_detector=lambda tail: ("en", 0.42, {"en": 0.42}),
            min_english_score=0.55,
        )
        self.assertEqual(result["context"], "")
        self.assertEqual(result["source"], "whisper_tail_language_uncertain")
        self.assertEqual(result["language_score"], 0.42)

    def test_whisper_tail_probe_skips_very_short_audio(self):
        class FakeWhisper:
            def transcribe(self, audio, **kwargs):
                raise AssertionError("Whisper should not be called")

        result = infer_hinglish_context_from_audio_tail(
            [0.0] * 8000, FakeWhisper()
        )
        self.assertEqual(result["context"], "")
        self.assertEqual(result["source"], "whisper_tail_too_short")

    def test_empty_input_returns_no_context(self):
        self.assertEqual(build_hinglish_context(""), "")


if __name__ == "__main__":
    unittest.main()
