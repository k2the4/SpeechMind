import unittest
from unittest.mock import Mock, patch

from models.wsl_translation_provider import WslIndicTransProvider


class WslTranslationCacheTests(unittest.TestCase):
    def test_repeated_translation_uses_process_cache(self):
        provider = WslIndicTransProvider(project_root=".")
        request = Mock(return_value={
            "ok": True,
            "translated_text": "I am not sure about the layout.",
            "model": "ai4bharat/indictrans2-indic-en-dist-200M",
            "inference_seconds": 12.5,
            "initialization_seconds": 6.0,
            "preparation_seconds": 0.5,
            "generation_seconds": 4.0,
            "postprocessing_seconds": 0.25,
            "worker_request_seconds": 12.5,
        })
        provider._request = request

        first = provider.translate_to_english("मुझे layout चाहिए", "hin_Deva")
        second = provider.translate_to_english("मुझे layout चाहिए", "hin_Deva")

        self.assertEqual(first["translated_text"], second["translated_text"])
        self.assertFalse(first["cache_used"])
        self.assertEqual(first["initialization_seconds"], 6.0)
        self.assertEqual(first["generation_seconds"], 4.0)
        self.assertEqual(first["preparation_seconds"], 0.5)
        self.assertEqual(first["postprocessing_seconds"], 0.25)
        self.assertEqual(first["worker_request_seconds"], 12.5)
        self.assertTrue(second["cache_used"])
        self.assertEqual(second["inference_seconds"], 0.0)
        request.assert_called_once()

    def test_different_language_does_not_hit_cache(self):
        provider = WslIndicTransProvider(project_root=".")
        request = Mock(side_effect=[
            {"ok": True, "translated_text": "Hello", "inference_seconds": 1.0},
            {"ok": True, "translated_text": "Different", "inference_seconds": 1.0},
        ])
        provider._request = request

        provider.translate_to_english("hello", "hin_Deva")
        second = provider.translate_to_english("hello", "ben_Beng")

        self.assertFalse(second["cache_used"])
        self.assertEqual(request.call_count, 2)


    def test_cache_is_separate_for_each_beam_setting(self):
        provider = WslIndicTransProvider(project_root=".")
        request = Mock(side_effect=[
            {"ok": True, "translated_text": "Four beam", "num_beams": 4},
            {"ok": True, "translated_text": "Two beam", "num_beams": 2},
        ])
        provider._request = request

        with patch.dict("os.environ", {"SPEECHMIND_TRANSLATION_BEAMS": "4"}):
            four = provider.translate_to_english("same input", "hin_Deva")
        with unittest.mock.patch.dict("os.environ", {"SPEECHMIND_TRANSLATION_BEAMS": "2"}):
            two = provider.translate_to_english("same input", "hin_Deva")

        self.assertEqual(four["num_beams"], 4)
        self.assertEqual(two["num_beams"], 2)
        self.assertFalse(two["cache_used"])
        self.assertEqual(request.call_count, 2)



if __name__ == "__main__":
    unittest.main()
