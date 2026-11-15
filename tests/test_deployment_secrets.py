import os
import sys
import types
import unittest
from unittest import mock

from models import deployment_checkpoints as checkpoints


class DeploymentSecretTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("SPEECHMIND_MODEL_REPO", None)
        sys.modules.pop("streamlit", None)

    def test_environment_values_take_precedence(self):
        os.environ["HF_TOKEN"] = "environment-token"
        os.environ["SPEECHMIND_MODEL_REPO"] = "owner/environment-models"
        fake_streamlit = types.SimpleNamespace(
            secrets={
                "HF_TOKEN": "streamlit-token",
                "SPEECHMIND_MODEL_REPO": "owner/streamlit-models",
            }
        )
        with mock.patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            self.assertEqual(checkpoints._deployment_setting("HF_TOKEN"), "environment-token")
            self.assertEqual(
                checkpoints._deployment_setting("SPEECHMIND_MODEL_REPO"),
                "owner/environment-models",
            )

    def test_reads_streamlit_secrets_when_environment_is_empty(self):
        fake_streamlit = types.SimpleNamespace(
            secrets={
                "HF_TOKEN": "streamlit-token",
                "SPEECHMIND_MODEL_REPO": "owner/speechmind-models",
            }
        )
        with mock.patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            self.assertEqual(checkpoints._deployment_setting("HF_TOKEN"), "streamlit-token")
            self.assertEqual(
                checkpoints._deployment_setting("SPEECHMIND_MODEL_REPO"),
                "owner/speechmind-models",
            )

    def test_configure_huggingface_auth_exports_streamlit_token(self):
        fake_streamlit = types.SimpleNamespace(
            secrets={"HF_TOKEN": "streamlit-token"}
        )
        with mock.patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            self.assertEqual(
                checkpoints.configure_huggingface_auth(), "streamlit-token"
            )
        self.assertEqual(os.environ["HF_TOKEN"], "streamlit-token")


if __name__ == "__main__":
    unittest.main()
