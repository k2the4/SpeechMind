"""Optional checkpoint bootstrap for hosted SpeechMind deployments.

Training checkpoints are intentionally excluded from Git. A deployment can
pull them from a private or public Hugging Face model repository by setting
SPEECHMIND_MODEL_REPO and, for private repositories, HF_TOKEN.
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def ensure_deployment_checkpoints() -> bool:
    repo_id = os.getenv("SPEECHMIND_MODEL_REPO", "").strip()
    if not repo_id:
        return False

    required = [
        _ROOT / "checkpoints" / "text_emotion_muril" / "speechmind_metadata.json",
        _ROOT / "checkpoints" / "voice_emotion_multicorpus_cpu" / "speechmind_metadata.json",
        _ROOT / "checkpoints" / "voice_emotion_multicorpus_cpu" / "voice_emotion_head.joblib",
    ]
    if all(path.is_file() for path in required):
        return True

    token = os.getenv("HF_TOKEN") or None
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
            f"{repo_id!r}. Check SPEECHMIND_MODEL_REPO and the HF_TOKEN "
            "read permission for a private model repository."
        ) from error

    missing = [str(path.relative_to(_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "The checkpoint repository is missing required files: "
            + ", ".join(missing)
        )
    return True
