import unittest
from pathlib import Path


class PipelineImportSafetyTests(unittest.TestCase):
    def test_torch_runtime_is_configured_inside_pipeline_constructor(self):
        source = (
            Path(__file__).resolve().parents[1] / "pipeline.py"
        ).read_text(encoding="utf-8")
        class_line = source.index("class SpeechMindPipeline:")
        configure_line = source.index(
            "self.torch_runtime = configure_torch_runtime()"
        )
        self.assertGreater(configure_line, class_line)
        self.assertNotIn("TORCH_RUNTIME_CONFIG = configure_torch_runtime()", source)


if __name__ == "__main__":
    unittest.main()
