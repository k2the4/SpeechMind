import ast
import unittest
from pathlib import Path


class BenchmarkEntrypointTests(unittest.TestCase):
    def test_repository_root_is_added_before_project_package_imports(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "evaluate_indicvoices_asr.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)

        root_assignment_line = None
        path_insert_line = None
        first_project_import_line = None

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if any(
                    isinstance(target, ast.Name) and target.id == "PROJECT_ROOT"
                    for target in node.targets
                ):
                    root_assignment_line = node.lineno
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if (
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "insert"
                    and isinstance(call.func.value, ast.Attribute)
                    and call.func.value.attr == "path"
                ):
                    path_insert_line = node.lineno
            elif isinstance(node, ast.ImportFrom) and node.module in {
                "models.asr",
                "scripts.asr_metrics",
                "utils.audio_utils",
            }:
                first_project_import_line = min(
                    first_project_import_line or node.lineno, node.lineno
                )

        self.assertIsNotNone(root_assignment_line)
        self.assertIsNotNone(path_insert_line)
        self.assertIsNotNone(first_project_import_line)
        self.assertLess(root_assignment_line, first_project_import_line)
        self.assertLess(path_insert_line, first_project_import_line)
        self.assertIn(
            'DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "data/indicvoices/hindi/valid_sample"',
            source,
        )



    def test_indicconformer_warmup_is_before_clip_timing(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "evaluate_indicvoices_asr.py"
        ).read_text(encoding="utf-8")
        self.assertIn('if args.model in ("auto", "indicconformer"):', source)
        warmup_line = source.index("recognizer._load_indic_model()")
        inference_timer_line = source.index("inference_started = time.perf_counter()")
        loop_line = source.index("for index, record in enumerate(records, start=1):")
        self.assertLess(warmup_line, inference_timer_line)
        self.assertLess(inference_timer_line, loop_line)
        self.assertIn('"model_initialization_seconds"', source)
        self.assertIn('"median_inference_seconds"', source)

    def test_forced_indicconformer_launcher_exists(self):
        root = Path(__file__).resolve().parents[1]
        launcher = root / "BENCHMARK_INDICVOICES_HINDI_INDICCONFORMER.bat"
        self.assertTrue(launcher.is_file())
        content = launcher.read_text(encoding="utf-8")
        self.assertIn("--model indicconformer", content)


    def test_benchmark_reports_script_and_duration_diagnostics(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "evaluate_indicvoices_asr.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def detect_script(", source)
        self.assertIn("def duration_bucket(", source)
        self.assertIn('"hypothesis_script": output_script', source)
        self.assertIn('"hypothesis_script_counts": dict(script_counts)', source)
        self.assertIn('"by_duration":', source)

    def test_forced_whisper_launcher_exists(self):
        root = Path(__file__).resolve().parents[1]
        launcher = root / "BENCHMARK_INDICVOICES_HINDI_WHISPER.bat"
        self.assertTrue(launcher.is_file())
        self.assertIn(
            "--model whisper",
            launcher.read_text(encoding="utf-8"),
        )

if __name__ == "__main__":
    unittest.main()
