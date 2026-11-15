import json
import sys
import types
import unittest
from unittest.mock import patch

import local_llm


class _FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class LocalLLMTests(unittest.TestCase):
    def test_gemini_key_is_read_from_environment_at_call_time(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "  test-key  "}):
            self.assertEqual(local_llm._get_gemini_api_key(), "test-key")

    def test_gemini_key_can_be_read_from_streamlit_secrets(self):
        fake_streamlit = types.SimpleNamespace(secrets={"GEMINI_API_KEY": "secret-key"})
        with patch.dict("os.environ", {}, clear=True), patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            self.assertEqual(local_llm._get_gemini_api_key(), "secret-key")

    @patch("local_llm.urllib.request.urlopen")
    def test_gemini_uses_api_key_header_and_current_flash_lite_model(self, mocked_urlopen):
        mocked_urlopen.return_value = _FakeResponse({"candidates": [{"content": {"parts": [{"text": json.dumps({"ok": True})}]}}]})
        with patch("local_llm._get_gemini_api_key", return_value="test-key"):
            result = local_llm._post_ollama({"messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]})
        request = mocked_urlopen.call_args.args[0]
        self.assertIn("gemini-3.5-flash-lite:generateContent", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "test-key")
        self.assertNotIn("key=", request.full_url)
        self.assertEqual(json.loads(result["message"]["content"]), {"ok": True})

    def test_extracts_amount_from_hindi_transcript_without_trusting_model(self):
        transcript = "मेरे 2500 रुपये कट गए और order confirm नहीं हुआ।"
        result = local_llm._extract_grounded_details(transcript)
        self.assertEqual(result["amount"], "2500")

    def test_extracts_spoken_hindi_amount_pachis_sau(self):
        result = local_llm._extract_grounded_details(
            "मैंने पच्चीस सौ रुपये का पेमेंट किया था।"
        )
        self.assertEqual(result["amount"], "2500")

    def test_extracts_spoken_hindi_amount_two_thousand_five_hundred(self):
        result = local_llm._extract_grounded_details(
            "दो हजार पाँच सौ रुपये कट गए।"
        )
        self.assertEqual(result["amount"], "2500")

    def test_evidence_must_be_verbatim(self):
        transcript = "मेरे खाते से 2500 रुपये कट गए।"
        self.assertTrue(local_llm._evidence_is_verbatim("2500 रुपये कट गए", transcript))
        self.assertFalse(local_llm._evidence_is_verbatim("पैसे वापस नहीं मिले", transcript))

    @patch("local_llm.urllib.request.urlopen")
    def test_local_llm_output_is_validated_and_structured(self, mocked_urlopen):
        transcript = "मेरे खाते से 2500 रुपये कट गए और order confirm नहीं हुआ।"
        model_output = {
            "english_meaning": "2500 rupees were deducted but the order was not confirmed.",
            "primary_category": "Payments & refunds",
            "summary": "The speaker reports a deducted payment and an unconfirmed order.",
            "topics": [
                {
                    "category": "Payments & refunds",
                    "summary": "Payment deducted.",
                    "evidence": ["मेरे खाते से 2500 रुपये कट गए"],
                },
                {
                    "category": "Account access",
                    "summary": "Unsupported false positive.",
                    "evidence": ["account access problem"],
                },
            ],
            "signals": [
                {
                    "label": "Possible unresolved issue",
                    "description": "No resolution is stated.",
                    "evidence": ["order confirm नहीं हुआ"],
                },
                {
                    "label": "Possible escalation",
                    "description": "Fabricated.",
                    "evidence": ["manager को escalate करें"],
                },
            ],
            "missing_information": ["Can you share the order ID?"],
            "suggested_next_step": "Verify the payment and order status.",
            "suggested_status": "review_required",
        }
        mocked_urlopen.return_value = _FakeResponse({
            "message": {"content": json.dumps(model_output, ensure_ascii=False)}
        })
        result = local_llm.analyze_with_local_llm(transcript, language="hi")
        self.assertEqual(result["case_brief"]["primary_category"], "Payments & refunds")
        self.assertEqual(result["case_brief"]["reported_details"]["amount"]["value"], "2500")
        self.assertEqual(len(result["topics"]), 1)
        self.assertEqual(len(result["signals"]), 1)
        self.assertEqual(result["signals"][0]["label"], "Possible unresolved issue")
        self.assertEqual(result["english_meaning"], model_output["english_meaning"])

    @patch("local_llm.urllib.request.urlopen")
    def test_invalid_model_json_raises_for_safe_fallback(self, mocked_urlopen):
        mocked_urlopen.return_value = _FakeResponse({
            "message": {"content": "not json"}
        })
        with self.assertRaises(json.JSONDecodeError):
            local_llm.analyze_with_local_llm("Hello")


if __name__ == "__main__":
    unittest.main()
