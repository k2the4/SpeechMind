import sys
import tempfile
import unittest
from pathlib import Path

from models.srota_provider import SrotaASRProvider, has_mixed_devanagari_latin


class SrotaScriptDetectionTests(unittest.TestCase):
    def test_detects_hindi_english_code_switch(self):
        self.assertTrue(has_mixed_devanagari_latin("मेरा favourite colour blue है"))

    def test_does_not_classify_pure_hindi_as_mixed(self):
        self.assertFalse(has_mixed_devanagari_latin("मुझे यह कहानी पसंद है"))

    def test_does_not_classify_pure_english_as_mixed(self):
        self.assertFalse(has_mixed_devanagari_latin("I like this story"))




class SrotaModelSelectionTests(unittest.TestCase):
    def test_general_model_remains_the_default(self):
        provider = SrotaASRProvider()
        self.assertEqual(provider.model_name, SrotaASRProvider.MODEL_NAME)

    def test_conversational_specialist_can_be_selected(self):
        provider = SrotaASRProvider(
            model_name=SrotaASRProvider.CONVERSATIONAL_MODEL_NAME
        )
        self.assertEqual(
            provider.model_name, SrotaASRProvider.CONVERSATIONAL_MODEL_NAME
        )

class SrotaWorkerProtocolTests(unittest.TestCase):
    def test_worker_is_persistent_and_uses_jsonl_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = root / "scripts" / "srota_asr_worker.py"
            script.parent.mkdir(parents=True)
            script.write_text(
                "import json, sys\n"
                "print(json.dumps({'ready': True}), flush=True)\n"
                "count = 0\n"
                "for line in sys.stdin:\n"
                " request = json.loads(line)\n"
                " count += 1\n"
                " if request.get('action') == 'translate':\n"
                "  response = {'ok': True, 'translated_text': 'This is our thought about this image.', 'count': count}\n"
                " else:\n"
                "  response = {'ok': True, 'transcript': 'मेरा image है', 'count': count, 'context': request.get('context'), 'model_name': request.get('model_name')}\n"
                " print(json.dumps(response, ensure_ascii=True), flush=True)\n",
                encoding="utf-8",
            )
            audio = root / "sample.wav"
            audio.write_bytes(b"placeholder")
            provider = SrotaASRProvider(
                root,
                startup_timeout=5,
                request_timeout=5,
                python_executable=sys.executable,
                worker_script=script,
                model_name=SrotaASRProvider.CONVERSATIONAL_MODEL_NAME,
            )
            try:
                first = provider.transcribe_file(audio)
                second = provider.transcribe_file(audio, context="layout, Figma")
                self.assertEqual(first["transcript"], "मेरा image है")
                self.assertEqual(first["asr_model"], provider.CONVERSATIONAL_MODEL_NAME)
                self.assertEqual(
                    first["worker_response"]["model_name"], provider.CONVERSATIONAL_MODEL_NAME
                )
                self.assertEqual(second["transcript"], "मेरा image है")
                self.assertEqual(second["worker_response"]["count"], 2)
                self.assertIn("Possible names or technical terms: layout, Figma", second["worker_response"]["context"])
                self.assertIn("natural Hindi-English", first["worker_response"]["context"])
                translation = provider.translate_to_english(
                    "इस image के बारे में हमारा ये thought है", "hin_Deva"
                )
                self.assertEqual(
                    translation["translated_text"],
                    "This is our thought about this image.",
                )
                self.assertEqual(translation["worker_response"]["count"], 3)
            finally:
                provider.close()


if __name__ == "__main__":
    unittest.main()
