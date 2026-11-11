"""Smoke-test IndicTrans2 through the isolated SpeechMind model worker."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.translation import Translator


def main() -> int:
    original = "इस image के बारे में हमारा ये thought है"
    print("[Translation smoke] Original Hinglish transcript:")
    print(original)
    try:
        translator = Translator()
        translated = translator.to_english(original, "hi")
    except Exception as error:
        print(f"\n[Translation smoke] FAILED: {type(error).__name__}: {error}")
        return 1

    print("\n[Translation smoke] English meaning:")
    print(translated)
    if not translated or translated == original:
        print("\n[Translation smoke] FAILED: no distinct English meaning was returned.")
        return 1

    print("\n[Translation smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
