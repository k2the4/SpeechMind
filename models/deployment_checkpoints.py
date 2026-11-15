"""Optional checkpoint bootstrap for hosted SpeechMind deployments.

Training checkpoints are intentionally excluded from Git. A deployment can
pull them from a private or public Hugging Face model repository by setting
SPEECHMIND_MODEL_REPO and, for private repositories, HF_TOKEN.
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


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

    # Hugging Face sequence-classification checkpoints may use either safetensors
    # or the legacy PyTorch weight filename.
    if not any((text_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        missing.append("checkpoints/text_emotion_muril/(model.safetensors or pytorch_model.bin)")
    if not any((text_dir / name).is_file() for name in ("tokenizer.json", "spiece.model", "vocab.txt")):
        missing.append("checkpoints/text_emotion_muril/(tokenizer.json, spiece.model or vocab.txt)")
    return missing


def ensure_deployment_checkpoints() -> bool:
    repo_id = os.getenv("SPEECHMIND_MODEL_REPO", "").strip()
    missing = _missing_checkpoint_files()
    if not missing:
        return True
    if not repo_id:
        print(
            "[Deployment] Trained checkpoints are not configured. Set "
            "SPEECHMIND_MODEL_REPO (and HF_TOKEN for a private repo) to enable "
            "fine-tuned MuRIL and the trained voice-emotion head. Missing: "
            + ", ".join(missing)
        )
        return False

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

    missing = _missing_checkpoint_files()
    if missing:
        raise RuntimeError(
            "The checkpoint repository is missing required files: "
            + ", ".join(missing)
        )
    return True
