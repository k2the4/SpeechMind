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


    def test_two_beam_mode_is_supported_without_enabling_kv_cache(self):
        model = FakeModel()
        generate_translation_without_cache(
            model, {"input_ids": [1]}, FakeTorch(), num_beams=2
        )
        self.assertEqual(model.calls[0]["num_beams"], 2)
        self.assertFalse(model.calls[0]["use_cache"])

    def test_unsupported_beam_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_beams"):
            generate_translation_without_cache(
                FakeModel(), {"input_ids": [1]}, FakeTorch(), num_beams=3
            )



    def test_worker_reports_model_initialization_separately_from_generation(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "scripts" / "indictrans_wsl_worker.py"
        ).read_text(encoding="utf-8")
        function_start = source.index("def translate_to_english(")
        main_start = source.index("def main()", function_start)
        function = source[function_start:main_start]
        init_end = function.index("initialization_seconds = time.perf_counter() - initialization_started")
        prep_start = function.index("preparation_started = time.perf_counter()")
        generation_start = function.index("generation_started = time.perf_counter()")
        generation_end = function.index("generation_seconds = time.perf_counter() - generation_started")
        post_start = function.index("postprocessing_started = time.perf_counter()")
        self.assertLess(init_end, prep_start)
        self.assertLess(prep_start, generation_start)
        self.assertLess(generation_start, generation_end)
        self.assertLess(generation_end, post_start)
        self.assertIn('"initialization_seconds": round(initialization_seconds, 3)', function)
        self.assertIn('"generation_seconds": round(generation_seconds, 3)', function)


if __name__ == "__main__":
    unittest.main()
