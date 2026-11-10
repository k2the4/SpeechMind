import ast
import unittest
from pathlib import Path


class IndicConformerFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.source = (root / "models" / "asr.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.requirements = (root / "requirements.txt").read_text(encoding="utf-8")

    def test_cpu_onnx_runtime_is_declared(self):
        self.assertIn("onnx==1.20.1", self.requirements)
        self.assertIn("onnxruntime==1.20.1", self.requirements)
        self.assertNotIn("onnxruntime-gpu", self.requirements)

    def test_indicconformer_initialization_failure_is_cached(self):
        self.assertIn("self.indic_model_error = error", self.source)
        self.assertIn("IndicConformer initialization previously failed", self.source)

    def test_indicconformer_failure_uses_whisper_fallback(self):
        fallback_methods = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_indic_result_with_whisper_fallback"
        ]
        self.assertEqual(len(fallback_methods), 1)
        method = fallback_methods[0]
        has_handler = any(
            isinstance(node, ast.Try) and bool(node.handlers)
            for node in ast.walk(method)
        )
        calls_whisper = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_decode_with_whisper"
            for node in ast.walk(method)
        )
        records_fallback = "asr_fallback_reason" in self.source
        self.assertTrue(has_handler)
        self.assertTrue(calls_whisper)
        self.assertTrue(records_fallback)

    def test_automatic_indic_route_uses_fallback_wrapper(self):
        self.assertIn(
            "return self._indic_result_with_whisper_fallback(",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
