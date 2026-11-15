"""Deployment-aware defaults for optional isolated model workers."""
from __future__ import annotations

import os
from pathlib import Path


def has_isolated_srota_runtime(project_root=None, is_windows=None):
    """Return whether the isolated local Srota worker environment is installed."""
    root = Path(project_root or Path(__file__).resolve().parents[1])
    windows = os.name == "nt" if is_windows is None else bool(is_windows)
    python = (
        root / ".venv_srota" / "Scripts" / "python.exe"
        if windows
        else root / ".venv_srota" / "bin" / "python"
    )
    worker = root / "scripts" / "srota_asr_worker.py"
    return python.is_file() and worker.is_file()


def has_remote_srota_runtime() -> bool:
    """Return whether a remote Srota endpoint and its API token are configured."""
    try:
        from models.deployment_checkpoints import _deployment_setting
        endpoint = _deployment_setting("SPEECHMIND_SROTA_ENDPOINT")
        token = _deployment_setting("SPEECHMIND_SROTA_API_TOKEN")
        return bool(endpoint and token)
    except Exception:
        return False


def default_analysis_mode_index(project_root=None, is_windows=None):
    """Prefer full analysis when either local or remote Srota is configured."""
    configured = (
        has_isolated_srota_runtime(project_root, is_windows)
        or has_remote_srota_runtime()
    )
    return 0 if configured else 1
