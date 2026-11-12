import unittest

from models.hinglish_context import (
    build_hinglish_context,
    extract_hinglish_context_terms,
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

    def test_empty_input_returns_no_context(self):
        self.assertEqual(build_hinglish_context(""), "")


if __name__ == "__main__":
    unittest.main()
