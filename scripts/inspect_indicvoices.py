"""Inspect a tiny streamed sample of the gated IndicVoices Hindi validation split.

This script does not download the full split or save audio. It reads at most
three records so we can verify access, schema, transcript fields, and audio
payload availability before building the real ASR benchmark loader.
"""
from __future__ import annotations

import sys

from datasets import Audio, load_dataset


REPOSITORY = "ai4bharat/IndicVoices"
CONFIG = "hindi"
SPLIT = "valid"
MAX_RECORDS = 3


def main() -> int:
    print(f"Opening {REPOSITORY} / {CONFIG} / {SPLIT} in streaming mode...")
    print("Only the first three records will be inspected; no bulk download.")

    try:
        stream = load_dataset(
            REPOSITORY,
            CONFIG,
            split=SPLIT,
            streaming=True,
            token=True,
        )
        stream = stream.cast_column("audio_filepath", Audio(decode=False))

        for index, row in enumerate(stream, start=1):
            audio = row.get("audio_filepath") or {}
            audio_bytes = audio.get("bytes")
            print(f"\n--- Record {index} ---")
            for field in (
                "text", "normalized", "verbatim", "duration", "lang",
                "speaker_id", "scenario", "task_name", "gender", "age_group",
                "state", "district",
            ):
                if field in row:
                    value = row[field]
                    if isinstance(value, str) and len(value) > 240:
                        value = value[:240] + "..."
                    print(f"{field}: {value}")
            print(f"audio_path: {audio.get('path')}")
            print(
                "audio_bytes: "
                + (str(len(audio_bytes)) if audio_bytes is not None else "not included in row")
            )
            if index >= MAX_RECORDS:
                break

        print("\nInspection complete. No dataset files were saved.")
        return 0

    except Exception as error:
        print("\nIndicVoices inspection failed.")
        print(f"{type(error).__name__}: {error}")
        print(
            "\nCheck that Hugging Face access is approved and that you have "
            "authenticated in this PC's Python environment with hf auth login."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
