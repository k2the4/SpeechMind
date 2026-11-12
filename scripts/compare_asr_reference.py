"""Compare Whisper and Srota on the same recording against a human reference.

Run from the project root:
  .venv\\Scripts\\python.exe scripts\\compare_asr_reference.py --audio "hinglish 2.wav"

The script deliberately bypasses automatic routing so both recognizers see the
same preprocessed waveform. It does not change production routing or prompts.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
import sys

# When a script is launched by path, Python puts scripts/ rather than the
# repository root on sys.path. Add the project root so package imports work
# from a normal Windows PowerShell invocation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import whisper

from models.srota_provider import SrotaASRProvider
from utils.audio_utils import preprocess


DEFAULT_REFERENCE = (
    "मुझे इस project के लिए एक नया डिजाइन बनाना है "
    "but i am not sure about the layout"
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or "")).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_token in enumerate(reference, start=1):
        current = [row]
        for col, hyp_token in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[col] + 1,
                previous[col - 1] + (ref_token != hyp_token),
            ))
        previous = current
    return previous[-1]


def error_rates(reference: str, hypothesis: str) -> dict[str, float]:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    ref_words = ref.split()
    hyp_words = hyp.split()
    wer = edit_distance(ref_words, hyp_words) / max(1, len(ref_words))
    cer = edit_distance(list(ref), list(hyp)) / max(1, len(ref))
    return {"WER": wer, "CER": cer}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=False, help="Path to the reference WAV/audio file")
    parser.add_argument(
        "--reference", default=DEFAULT_REFERENCE,
        help="Correct human transcript (defaults to the hinglish 2.wav reference)",
    )
    parser.add_argument(
        "--context", default=None,
        help="Optional identical task context for Srota only; omit for the baseline comparison",
    )
    args = parser.parse_args()

    audio_value = args.audio or input("Enter the audio file path: ").strip().strip(\'"\')
    if not audio_value:
        parser.error("An audio file path is required.")
    audio_path = Path(audio_value).expanduser().resolve()
    if not audio_path.is_file():
        parser.error(f"Audio file not found: {audio_path}")

    print(f"[ASR benchmark] Audio: {audio_path}")
    print("[ASR benchmark] Loading the same 16 kHz mono waveform for both models...")
    audio = preprocess(str(audio_path))
    if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.size == 0:
        raise RuntimeError("Audio preprocessing did not produce a non-empty mono waveform.")

    print("[ASR benchmark] Running Whisper-small (automatic language)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model("small", device=device)
    whisper_result = whisper_model.transcribe(
        audio, task="transcribe", language=None, fp16=False, verbose=False,
        beam_size=5, condition_on_previous_text=False,
    )
    whisper_text = str(whisper_result.get("text") or "").strip()

    print("[ASR benchmark] Running Srota/Qwen3-ASR...")
    srota = SrotaASRProvider()
    srota_result = srota.transcribe_audio(
        audio, sample_rate=16000, context=args.context
    )
    srota_text = str(srota_result.get("transcript") or "").strip()

    print("\n========== REFERENCE ==========")
    print(args.reference)
    print("\n========== WHISPER-SMALL ==========")
    print(whisper_text or "[empty transcript]")
    print(error_rates(args.reference, whisper_text))
    print("\n========== SROTA / QWEN3-ASR ==========")
    print(srota_text or "[empty transcript]")
    print(error_rates(args.reference, srota_text))
    print("\n[ASR benchmark] Lower WER/CER is better. Scores are diagnostic for this clip,")
    print("not a general model-quality claim. Production routing was not changed.")
    srota.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
