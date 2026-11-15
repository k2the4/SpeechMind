"""Deployment-aware defaults for optional isolated model workers."""
from __future__ import annotations

import os
from pathlib import Path


def has_isolated_srota_runtime(project_root=None, is_windows=None):
    """Return whether the isolated Srota worker environment is installed."""
    root = Path(project_root or Path(__file__).resolve().parents[1])
    windows = os.name == "nt" if is_windows is None else bool(is_windows)
    python = (
        root / ".venv_srota" / "Scripts" / "python.exe"
        if windows
        else root / ".venv_srota" / "bin" / "python"
    )
    worker = root / "scripts" / "srota_asr_worker.py"
    return python.is_file() and worker.is_file()


def default_analysis_mode_index(project_root=None, is_windows=None):
    """Prefer full analysis only when the isolated worker is present."""
    return 0 if has_isolated_srota_runtime(project_root, is_windows) else 1
