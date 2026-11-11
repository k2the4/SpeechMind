import os
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


class SrotaWorkerProtocolTests(unittest.TestCase):
    def test_worker_is_persistent_and_uses_jsonl_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            if os.name == "nt":
                python_path = root / ".venv_srota" / "Scripts" / "python.exe"
            else:
                python_path = root / ".venv_srota" / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.write_bytes(Path(sys.executable).read_bytes())
            script = root / "scripts" / "srota_asr_worker.py"
            script.parent.mkdir(parents=True)
            script.write_text(
                "import json, sys\n"
                "print(json.dumps({'ready': True}), flush=True)\n"
                "count = 0\n"
                "for line in sys.stdin:\n"
                " request = json.loads(line)\n"
                " count += 1\n"
                " print(json.dumps({'ok': True, 'transcript': 'मेरा image है', 'count': count}), flush=True)\n",
                encoding="utf-8",
            )
            audio = root / "sample.wav"
            audio.write_bytes(b"placeholder")
            provider = SrotaASRProvider(root, startup_timeout=5, request_timeout=5)
            try:
                first = provider.transcribe_file(audio)
                second = provider.transcribe_file(audio)
                self.assertEqual(first["transcript"], "मेरा image है")
                self.assertEqual(second["transcript"], "मेरा image है")
                self.assertEqual(second["count"], 2)
            finally:
                provider.close()


if __name__ == "__main__":
    unittest.main()
