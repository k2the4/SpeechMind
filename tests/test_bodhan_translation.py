import unittest
from unittest.mock import Mock, patch

from models.bodhan_translation import BodhanTranslator


class BodhanTranslatorTests(unittest.TestCase):
    @patch("models.bodhan_translation.requests.post")
    def test_translates_code_mixed_text_to_english(self, post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "The meeting has been canceled."}}]
        }
        post.return_value = response

        translator = BodhanTranslator(api_key="test-key")
        result = translator.to_english("Yaar tomorrow ka meeting cancel ho gaya hai.")

        self.assertEqual(result, "The meeting has been canceled.")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "indic-translate")
        self.assertEqual(payload["target_language_code"], "en")
        self.assertIn("Yaar tomorrow ka meeting", payload["messages"][0]["content"])
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key"
        )

    @patch("models.bodhan_translation.requests.post")
    def test_rejects_empty_translation(self, post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": " "}}]
        }
        post.return_value = response
        translator = BodhanTranslator(api_key="test-key")
        with self.assertRaisesRegex(RuntimeError, "empty translation"):
            translator.to_english("kal meeting hai")

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "BODHAN_API_KEY"):
                BodhanTranslator()


if __name__ == "__main__":
    unittest.main()
