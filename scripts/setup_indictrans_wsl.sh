#!/usr/bin/env bash
set -euo pipefail

VENV="$HOME/.venvs/speechmind-indictrans"
echo "[SpeechMind] Installing isolated IndicTrans2 runtime in WSL."
echo "[SpeechMind] This does not modify the Windows .venv or .venv_srota."

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip build-essential

mkdir -p "$HOME/.venvs"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5"
"$VENV/bin/python" -m pip install \
  "numpy>=2.1,<2.3" \
  "transformers==4.57.6" \
  "indictranstoolkit==1.1.1" \
  sentencepiece

"$VENV/bin/python" - <<'PY'
import torch
from transformers import AutoTokenizer
try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    from IndicTransToolkit import IndicProcessor
print("[SpeechMind] WSL imports: PASS")
print(f"[SpeechMind] PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
print("[SpeechMind] IndicProcessor:", IndicProcessor.__name__)
PY

echo
echo "[SpeechMind] WSL translation dependencies installed."
echo "[SpeechMind] The IndicTrans2 200M model will download on the first translation check."
