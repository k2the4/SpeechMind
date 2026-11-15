import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from models.deployment_policy import default_analysis_mode_index, has_isolated_srota_runtime


class DeploymentPolicyTests(unittest.TestCase):
    def test_hosted_runtime_without_worker_defaults_to_fast_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(has_isolated_srota_runtime(directory, is_windows=False))
            self.assertEqual(default_analysis_mode_index(directory, is_windows=False), 1)

    def test_linux_worker_presence_enables_full_mode_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / '.venv_srota' / 'bin' / 'python'
            worker = root / 'scripts' / 'srota_asr_worker.py'
            python.parent.mkdir(parents=True)
            worker.parent.mkdir(parents=True)
            python.touch()
            worker.touch()
            self.assertTrue(has_isolated_srota_runtime(root, is_windows=False))
            self.assertEqual(default_analysis_mode_index(root, is_windows=False), 0)


    def test_remote_worker_configuration_enables_full_mode_default(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {
                "SPEECHMIND_SROTA_ENDPOINT": "https://speechmind-srota.hf.space",
                "SPEECHMIND_SROTA_API_TOKEN": "test-token",
            }):
                self.assertEqual(
                    default_analysis_mode_index(directory, is_windows=False), 0
                )

    def test_windows_worker_presence_enables_full_mode_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / '.venv_srota' / 'Scripts' / 'python.exe'
            worker = root / 'scripts' / 'srota_asr_worker.py'
            python.parent.mkdir(parents=True)
            worker.parent.mkdir(parents=True)
            python.touch()
            worker.touch()
            self.assertTrue(has_isolated_srota_runtime(root, is_windows=True))
            self.assertEqual(default_analysis_mode_index(root, is_windows=True), 0)


if __name__ == '__main__':
    unittest.main()
