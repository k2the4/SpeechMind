"""Smoke-test IndicTrans2 through the isolated SpeechMind model worker."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.translation import Translator


MODEL_ACCESS_URL = (
    "https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M"
)


def main() -> int:
    original = "मुझे इस project के लिए एक नया डिजाइन बनाना है but i am not sure about the layout"
    try:
        translator = Translator()
        print(f"[Translation smoke] Provider: {translator.provider_name}")
        print("[Translation smoke] Original Hinglish transcript:")
        print(original)
        translated = translator.to_english(original, "hi")
        if translator.last_cache_used is not False:
            raise RuntimeError(
                "Expected the known-compatible IndicTrans2 no-cache path."
            )
    except Exception as error:
        details = f"{type(error).__name__}: {error}"
        print(f"\n[Translation smoke] FAILED: {details}")
        if "gated repo" in details.lower() or "401 client error" in details.lower():
            print("\n[Translation smoke] Hugging Face access is required.")
            print(f"1. Open {MODEL_ACCESS_URL}")
            print("2. Sign in and accept/request access to the model repository.")
            print("3. Authenticate inside WSL using: hf auth login")
            print("   Run it in the IndicTrans WSL environment, not Windows.")
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
