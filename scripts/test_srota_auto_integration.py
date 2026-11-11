"""One-clip end-to-end smoke test for automatic Srota code-switch routing."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.asr import SpeechRecognizer
from utils.audio_utils import load_audio


def main() -> int:
    audio_path = (
        PROJECT_ROOT
        / "data"
        / "hiacc"
        / "raw"
        / "Corpus"
        / "children"
        / "audio"
        / "test_split"
        / "CH21075.wav"
    )
    if not audio_path.is_file():
        print(f"[Srota smoke] Required prepared HiACC clip not found: {audio_path}")
        print("Run this check only on the PC where HiACC has already been prepared.")
        return 2

    worker_python = PROJECT_ROOT / ".venv_srota" / "Scripts" / "python.exe"
    if not worker_python.is_file():
        print("[Srota smoke] .venv_srota is missing. Run SETUP_HINGLISH_ASR_ENV.bat once.")
        return 2

    print("[Srota smoke] Loading SpeechMind ASR and testing one known Hinglish clip.")
    recognizer = SpeechRecognizer()
    audio = load_audio(str(audio_path))
    result = recognizer.transcribe(audio)

    print("\n[Srota smoke] Result")
    print(f"Detected primary language : {result.get('language')}")
    print(f"Selected ASR model         : {result.get('asr_model')}")
    print(f"Code-switch detected       : {result.get('code_switch_detected', False)}")
    print(f"Language source            : {result.get('language_source')}")
    print(f"Transcript                 : {result.get('transcript')}")

    if result.get("asr_model") != "moorlee/qwen3-asr-0.6b-hinglish":
        print("\n[Srota smoke] The sample did not select Srota. Review the routing logs above.")
        return 1
    if not result.get("code_switch_detected"):
        print("\n[Srota smoke] Srota ran but the output was not classified as mixed-script.")
        return 1

    print("\n[Srota smoke] PASS: isolated worker and automatic mixed-script route are connected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
