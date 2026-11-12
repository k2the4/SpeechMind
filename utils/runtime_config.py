"""Conservative CPU runtime settings for local SpeechMind inference.

The main app runs several CPU-backed models concurrently. Bounding PyTorch's
intra-op thread pool avoids each branch attempting to occupy every core.
Set SPEECHMIND_TORCH_THREADS to override the default (4).
"""
from __future__ import annotations

import os


DEFAULT_TORCH_THREADS = 4
MAX_TORCH_THREADS = 16


def configure_torch_runtime(torch_module=None, environ=None, cpu_count=None) -> dict:
    """Configure PyTorch CPU thread pools before loading model weights.

    The helper is safe in lightweight environments where torch is not
    installed. Its dependencies are injectable so tests do not need PyTorch.
    """
    environment = os.environ if environ is None else environ
    try:
        requested = int(environment.get("SPEECHMIND_TORCH_THREADS", DEFAULT_TORCH_THREADS))
    except (TypeError, ValueError):
        requested = DEFAULT_TORCH_THREADS
    requested = max(1, min(requested, MAX_TORCH_THREADS))

    available_cpus = cpu_count if cpu_count is not None else (os.cpu_count() or requested)
    threads = max(1, min(requested, max(1, int(available_cpus))))

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return {
                "available": False,
                "reason": "torch_not_installed",
                "intra_op_threads": None,
                "interop_threads": None,
            }

    torch_module.set_num_threads(threads)
    interop_threads = max(1, min(2, threads))
    interop_configured = True
    try:
        torch_module.set_num_interop_threads(interop_threads)
    except RuntimeError:
        # PyTorch only permits this before parallel work starts. In an
        # embedding environment it may already be initialized; intra-op
        # configuration remains active and the process can continue safely.
        interop_configured = False

    try:
        device = "cuda" if torch_module.cuda.is_available() else "cpu"
    except (AttributeError, RuntimeError):
        device = "unknown"

    return {
        "available": True,
        "device": device,
        "intra_op_threads": threads,
        "interop_threads": interop_threads if interop_configured else None,
        "interop_configuration_skipped": not interop_configured,
    }
