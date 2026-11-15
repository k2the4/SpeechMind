"""Optional checkpoint bootstrap for hosted SpeechMind deployments.

Training checkpoints are intentionally excluded from Git. A deployment can
pull them from a private or public Hugging Face model repository by setting
SPEECHMIND_MODEL_REPO and, for private repositories, HF_TOKEN.

Streamlit Community Cloud stores these values in st.secrets rather than
exporting them as process environment variables. This module reads either
source and exports HF_TOKEN for Transformers/Hub model loaders.
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _deployment_setting(name: str) -> str:
    """Read a deployment setting from environment or Streamlit secrets."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st

        value = str(st.secrets.get(name, "") or "").strip()
    except Exception:
        # Streamlit is optional for local scripts/tests, and secrets may not
        # be configured. Do not make model initialization depend on st.secrets.
        value = ""
    return value


def configure_huggingface_auth() -> str | None:
    """Expose a Streamlit HF token to libraries that read HF_TOKEN from env."""
    token = _deployment_setting("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
    return token or None


def _missing_checkpoint_files() -> list[str]:
    text_dir = _ROOT / "checkpoints" / "text_emotion_muril"
    voice_dir = _ROOT / "checkpoints" / "voice_emotion_multicorpus_cpu"
    required = [
        text_dir / "speechmind_metadata.json",
        text_dir / "config.json",
        voice_dir / "speechmind_metadata.json",
        voice_dir / "voice_emotion_head.joblib",
    ]
    missing = [str(path.relative_to(_ROOT)) for path in required if not path.is_file()]

    if not any((text_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        missing.append("checkpoints/text_emotion_muril/(model.safetensors or pytorch_model.bin)")
    if not any((text_dir / name).is_file() for name in ("tokenizer.json", "spiece.model", "vocab.txt")):
        missing.append("checkpoints/text_emotion_muril/(tokenizer.json, spiece.model or vocab.txt)")
    return missing


def ensure_deployment_checkpoints() -> bool:
    # Configure authentication before any Transformers model is initialized.
    token = configure_huggingface_auth()
    repo_id = _deployment_setting("SPEECHMIND_MODEL_REPO")
    missing = _missing_checkpoint_files()
    if not missing:
        return True
    if not repo_id:
        print(
            "[Deployment] Trained checkpoints are not configured. Set "
            "SPEECHMIND_MODEL_REPO (and HF_TOKEN for a private repo) in the "
            "deployment environment or Streamlit secrets to enable fine-tuned "
            "MuRIL and the trained voice-emotion head. Missing: "
            + ", ".join(missing)
        )
        return False

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(_ROOT),
            allow_patterns=["checkpoints/**"],
            token=token,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not download SpeechMind model checkpoints from "
            f"{repo_id!r}. Verify the repository ID, that the token has read "
            "access to it, and that the repository contains the expected "
            "checkpoints/** directory."
        ) from error

    missing = _missing_checkpoint_files()
    if missing:
        raise RuntimeError(
            "The checkpoint repository is missing required files: "
            + ", ".join(missing)
        )
    return True
