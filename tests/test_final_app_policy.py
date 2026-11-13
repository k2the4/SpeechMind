import unittest
from pathlib import Path


class FinalAppPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.app = (root / "app.py").read_text(encoding="utf-8")

    def test_transcript_correction_controls_are_not_in_final_ui(self):
        self.assertNotIn("Review and correct transcript", self.app)
        self.assertNotIn("Apply correction and rerun", self.app)
        self.assertNotIn("speechmind_prepared", self.app)
        self.assertIn('st.subheader("Transcript (original speech)")', self.app)

    def test_complete_json_export_remains_available(self):
        self.assertIn("Download complete analysis (JSON)", self.app)
        self.assertIn('file_name="speechmind-analysis.json"', self.app)

    def test_app_uses_single_pass_pipeline_entrypoint(self):
        self.assertIn("result = pipeline.run(tmp_path)", self.app)
        self.assertNotIn("pipeline.prepare(tmp_path)", self.app)


if __name__ == "__main__":
    unittest.main()
