#!/usr/bin/env bash
set -euo pipefail

VENV="$HOME/.venvs/speechmind-indictrans"
UV_BOOTSTRAP="$HOME/.venvs/speechmind-uv-bootstrap"

echo "[SpeechMind] Installing isolated IndicTrans2 runtime in WSL."
echo "[SpeechMind] This does not modify the Windows .venv or .venv_srota."

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip build-essential

mkdir -p "$HOME/.venvs"

# Ubuntu 26.04 ships Python 3.14 by default. The pinned NumPy 2.2 range
# does not provide a cp314 wheel, so pip falls back to a source build.
# Bootstrap uv in its own environment and let it provision a wheel-supported
# CPython 3.11 runtime dedicated to IndicTrans2.
if [[ ! -x "$UV_BOOTSTRAP/bin/python" ]]; then
  python3 -m venv "$UV_BOOTSTRAP"
fi
"$UV_BOOTSTRAP/bin/python" -m pip install --upgrade pip uv
UV="$UV_BOOTSTRAP/bin/uv"

"$UV" python install 3.11
"$UV" venv --clear --python 3.11 "$VENV"

# Keep the CPU-only PyTorch wheel separate from the public PyPI packages.
"$UV" pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.5"
"$UV" pip install --python "$VENV/bin/python" \
  "numpy>=2.1,<2.3" \
  "transformers==4.57.6" \
  "indictranstoolkit==1.1.1" \
  sentencepiece

"$VENV/bin/python" - <<'PY'
import sys
import torch
import numpy
from transformers import AutoTokenizer
try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    from IndicTransToolkit import IndicProcessor

print("[SpeechMind] WSL imports: PASS")
print(f"[SpeechMind] Python: {sys.version.split()[0]}")
print(f"[SpeechMind] NumPy: {numpy.__version__}")
print(f"[SpeechMind] PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
print("[SpeechMind] IndicProcessor:", IndicProcessor.__name__)
PY

echo
echo "[SpeechMind] WSL translation dependencies installed."
echo "[SpeechMind] The IndicTrans2 200M model will download on the first translation check."
