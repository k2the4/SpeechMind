import unittest

from scripts.indictrans_wsl_worker import generate_translation_without_cache


class FakeTorch:
    @staticmethod
    def inference_mode():
        from contextlib import nullcontext
        return nullcontext()


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return ["generated"]


class TranslationWorkerGenerationTests(unittest.TestCase):
    def test_indictrans_generation_always_disables_cache(self):
        model = FakeModel()
        output = generate_translation_without_cache(
            model, {"input_ids": [1]}, FakeTorch()
        )

        self.assertEqual(output, ["generated"])
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(model.calls[0]["use_cache"])
        self.assertEqual(model.calls[0]["num_beams"], 4)
        self.assertEqual(model.calls[0]["max_length"], 256)
        self.assertEqual(model.calls[0]["num_return_sequences"], 1)

    def test_model_generation_errors_are_not_swallowed(self):
        class BrokenModel:
            def generate(self, **kwargs):
                raise RuntimeError("translation model failed")

        with self.assertRaisesRegex(RuntimeError, "translation model failed"):
            generate_translation_without_cache(
                BrokenModel(), {"input_ids": [1]}, FakeTorch()
            )


if __name__ == "__main__":
    unittest.main()
