import unittest

from scripts.indictrans_wsl_worker import generate_translation_with_cache


class FakeTorch:
    @staticmethod
    def inference_mode():
        from contextlib import nullcontext
        return nullcontext()


class FakeModel:
    def __init__(self, failure_message=None):
        self.failure_message = failure_message
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs["use_cache"])
        if self.failure_message and kwargs["use_cache"]:
            raise RuntimeError(self.failure_message)
        return ["generated"]


class TranslationWorkerCacheTests(unittest.TestCase):
    def test_cache_is_disabled_by_default_for_custom_indictrans_decoder(self):
        model = FakeModel()
        output, cache_used = generate_translation_with_cache(
            model, {"input_ids": [1]}, FakeTorch(), environ={}
        )
        self.assertEqual(output, ["generated"])
        self.assertFalse(cache_used)
        self.assertEqual(model.calls, [False])

    def test_cache_can_be_enabled_explicitly(self):
        model = FakeModel()
        output, cache_used = generate_translation_with_cache(
            model,
            {"input_ids": [1]},
            FakeTorch(),
            environ={"SPEECHMIND_INDIC_TRANS_USE_CACHE": "1"},
        )
        self.assertEqual(output, ["generated"])
        self.assertTrue(cache_used)
        self.assertEqual(model.calls, [True])

    def test_cache_can_be_disabled_explicitly(self):
        model = FakeModel()
        output, cache_used = generate_translation_with_cache(
            model,
            {"input_ids": [1]},
            FakeTorch(),
            environ={"SPEECHMIND_INDIC_TRANS_USE_CACHE": "0"},
        )
        self.assertEqual(output, ["generated"])
        self.assertFalse(cache_used)
        self.assertEqual(model.calls, [False])

    def test_cache_incompatibility_retries_without_cache(self):
        model = FakeModel("past_key_values is not supported")
        output, cache_used = generate_translation_with_cache(
            model,
            {"input_ids": [1]},
            FakeTorch(),
            environ={"SPEECHMIND_INDIC_TRANS_USE_CACHE": "1"},
        )
        self.assertEqual(output, ["generated"])
        self.assertFalse(cache_used)
        self.assertEqual(model.calls, [True, False])

    def test_unrelated_generation_error_is_not_hidden(self):
        model = FakeModel("out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            generate_translation_with_cache(
                model,
                {"input_ids": [1]},
                FakeTorch(),
                environ={"SPEECHMIND_INDIC_TRANS_USE_CACHE": "1"},
            )
        self.assertEqual(model.calls, [True])


if __name__ == "__main__":
    unittest.main()
