import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from models.srota_provider import SrotaASRProvider, has_mixed_devanagari_latin


class SrotaScriptDetectionTests(unittest.TestCase):
    def test_detects_hindi_english_code_switch(self):
        self.assertTrue(has_mixed_devanagari_latin("मेरा favourite colour blue है"))

    def test_does_not_classify_pure_hindi_as_mixed(self):
        self.assertFalse(has_mixed_devanagari_latin("मुझे यह कहानी पसंद है"))

    def test_does_not_classify_pure_english_as_mixed(self):
        self.assertFalse(has_mixed_devanagari_latin("I like this story"))




class SrotaLongAudioChunkingTests(unittest.TestCase):
    def test_long_audio_chunks_stay_under_model_limit_with_overlap(self):
        sample_rate = 16000
        waveform = np.zeros(int(32.2 * sample_rate), dtype=np.float32)
        ranges = SrotaASRProvider._chunk_audio(waveform, sample_rate)
        self.assertEqual(len(ranges), 2)
        self.assertTrue(all(
            end - start <= int(SrotaASRProvider.MAX_CLIP_SECONDS * sample_rate)
            for start, end in ranges
        ))
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[1][0], int(26 * sample_rate))
        self.assertEqual(ranges[-1][1], waveform.size)

    def test_chunk_transcripts_remove_repeated_overlap_words(self):
        merged = SrotaASRProvider._merge_chunk_transcripts([
            "कल हमारी team ने college project.",
            "ने college project, के लिए नया dashboard.",
        ])
        self.assertEqual(
            merged,
            "कल हमारी team ने college project, के लिए नया dashboard.",
        )

    def test_long_audio_is_transcribed_in_multiple_chunks(self):
        provider = SrotaASRProvider()
        provider.transcribe_file = mock.Mock(side_effect=[
            {
                "transcript": "कल हमारी team ने college project",
                "inference_seconds": 1.0,
                "worker_initialization_seconds": 2.0,
                "worker_request_seconds": 3.0,
                "worker_response": {"device": "cpu"},
            },
            {
                "transcript": "ने college project के लिए नया dashboard",
                "inference_seconds": 0.5,
                "worker_initialization_seconds": 0.0,
                "worker_request_seconds": 0.6,
                "worker_response": {"device": "cpu"},
            },
        ])
        try:
            result = provider.transcribe_audio(
                np.zeros(int(32.2 * 16000), dtype=np.float32),
                sample_rate=16000,
            )
        finally:
            provider.close()
        self.assertEqual(provider.transcribe_file.call_count, 2)
        self.assertEqual(result["chunk_count"], 2)
        self.assertEqual(
            result["transcript"],
            "कल हमारी team ने college project के लिए नया dashboard",
        )
        self.assertEqual(result["inference_seconds"], 1.5)
        self.assertEqual(result["worker_initialization_seconds"], 2.0)
        self.assertEqual(result["worker_request_seconds"], 3.6)



class SrotaTimeoutTests(unittest.TestCase):
    def test_timeout_stops_worker_and_preserves_stderr_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = root / "scripts" / "srota_asr_worker.py"
            script.parent.mkdir(parents=True)
            script.write_text(
                "import json, sys, time\n"
                "print(json.dumps({'ready': True}), flush=True)\n"
                "for line in sys.stdin:\n"
                " print('simulated model loading diagnostic', file=sys.stderr, flush=True)\n"
                " time.sleep(2)\n"
                " print(json.dumps({'ok': True, 'transcript': 'late response'}), flush=True)\n",
                encoding="utf-8",
            )
            audio = root / "sample.wav"
            audio.write_bytes(b"placeholder")
            provider = SrotaASRProvider(
                root,
                startup_timeout=2,
                request_timeout=0.1,
                python_executable=sys.executable,
                worker_script=script,
            )
            try:
                with self.assertRaisesRegex(TimeoutError, "stderr log"):
                    provider.transcribe_file(audio)
                self.assertIsNone(provider._process)
                self.assertTrue(provider.stderr_log_path.is_file())
                self.assertIn(
                    "simulated model loading diagnostic",
                    provider.stderr_log_path.read_text(encoding="utf-8"),
                )
            finally:
                provider.close()


class SrotaWorkerConfigTests(unittest.TestCase):
    def test_standard_qwen_generation_budget_is_preserved(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "srota_asr_worker.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"max_new_tokens": 512', source)
        self.assertIn("max_new_tokens=512", source)
        self.assertNotIn('"max_new_tokens": 256', source)


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
                "  response = {'ok': True, 'transcript': 'मेरा image है', 'count': count, 'context': request.get('context'), 'model_name': request.get('model_name'), 'inference_seconds': 1.25, 'worker_initialization_seconds': 2.5, 'worker_request_seconds': 3.75}\n"
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
                self.assertEqual(first["inference_seconds"], 1.25)
                self.assertEqual(first["worker_initialization_seconds"], 2.5)
                self.assertEqual(first["worker_request_seconds"], 3.75)
                self.assertEqual(first["asr_model"], provider.CONVERSATIONAL_MODEL_NAME)
                self.assertEqual(
                    first["worker_response"]["model_name"], provider.CONVERSATIONAL_MODEL_NAME
                )
                self.assertEqual(second["transcript"], "मेरा image है")
                self.assertEqual(second["worker_response"]["count"], 2)
                self.assertEqual(first["worker_response"]["context"], "")
                self.assertEqual(
                    second["worker_response"]["context"],
                    "layout, Figma",
                )
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



class SrotaWorkerTimingTests(unittest.TestCase):
    def test_model_initialization_is_outside_inference_timer(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "srota_asr_worker.py").read_text(
            encoding="utf-8"
        )
        start = source.index('if action == "transcribe":')
        end = source.index('elif action == "translate":', start)
        branch = source[start:end]
        init_end = branch.index("worker_initialization_seconds =")
        inference_timer = branch.index("inference_started = time.perf_counter()")
        model_call = branch.index("result = model.transcribe(")
        self.assertLess(init_end, inference_timer)
        self.assertLess(inference_timer, model_call)
        self.assertIn('"inference_seconds": round(inference_seconds, 3)', branch)
        self.assertIn('"worker_initialization_seconds": round(', branch)
        self.assertIn('"worker_request_seconds": round(worker_request_seconds, 3)', branch)


if __name__ == "__main__":
    unittest.main()
