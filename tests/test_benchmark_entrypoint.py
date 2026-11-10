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


if __name__ == "__main__":
    unittest.main()
