import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from models.sarvam_asr_provider import SarvamASRProvider


class FakeSpeechToText:
    def __init__(self):
        self.kwargs = None
        self.file_bytes = b""

    def transcribe(self, **kwargs):
        self.kwargs = kwargs
        self.file_bytes = kwargs["file"].read()
        return SimpleNamespace(
            transcript="मुझे इस project के लिए नया design चाहिए",
            language_code="hi-IN",
            request_id="test-request-123",
        )


class FakeClient:
    def __init__(self):
        self.speech_to_text = FakeSpeechToText()


class SarvamASRProviderTests(unittest.TestCase):
    def test_saaras_v4_uses_codemix_mode_and_returns_provider_metadata(self):
        client = FakeClient()
        provider = SarvamASRProvider(client=client)

        with tempfile.TemporaryDirectory() as temp:
            audio_path = Path(temp) / "clip.wav"
            sf.write(
                str(audio_path),
                np.zeros(16000, dtype=np.float32),
                16000,
                subtype="PCM_16",
            )
            result = provider.transcribe_file(
                audio_path,
                language_code="hi-IN",
                keyterms=["layout", "Figma"],
            )

        self.assertEqual(result["transcript"], "मुझे इस project के लिए नया design चाहिए")
        self.assertEqual(result["language"], "hi")
        self.assertEqual(result["asr_model"], "saaras:v4")
        self.assertEqual(result["asr_provider"], "sarvam")
        self.assertEqual(result["request_id"], "test-request-123")
        self.assertEqual(client.speech_to_text.kwargs["model"], "saaras:v4")
        self.assertEqual(client.speech_to_text.kwargs["mode"], "codemix")
        self.assertEqual(client.speech_to_text.kwargs["language_code"], "hi-IN")
        self.assertEqual(client.speech_to_text.kwargs["keyterms"], ["layout", "Figma"])
        self.assertTrue(client.speech_to_text.file_bytes.startswith(b"RIFF"))

    def test_missing_api_key_fails_with_setup_instruction(self):
        with self.assertRaisesRegex(RuntimeError, "SARVAM_API_KEY"):
            SarvamASRProvider(api_key="", client=None)

    def test_rest_clip_limit_is_checked_before_api_call(self):
        client = FakeClient()
        provider = SarvamASRProvider(client=client)

        with tempfile.TemporaryDirectory() as temp:
            audio_path = Path(temp) / "long.wav"
            sf.write(
                str(audio_path),
                np.zeros(31 * 16000, dtype=np.float32),
                16000,
                subtype="PCM_16",
            )
            with self.assertRaisesRegex(ValueError, "up to 30 seconds"):
                provider.transcribe_file(audio_path)

        self.assertIsNone(client.speech_to_text.kwargs)

    def test_empty_transcript_is_rejected(self):
        class EmptySpeechToText:
            def transcribe(self, **_kwargs):
                return SimpleNamespace(transcript="", language_code="hi-IN")

        client = SimpleNamespace(speech_to_text=EmptySpeechToText())
        provider = SarvamASRProvider(client=client)

        with tempfile.TemporaryDirectory() as temp:
            audio_path = Path(temp) / "clip.wav"
            sf.write(str(audio_path), np.zeros(16000, dtype=np.float32), 16000)
            with self.assertRaisesRegex(RuntimeError, "empty transcript"):
                provider.transcribe_file(audio_path)


if __name__ == "__main__":
    unittest.main()
