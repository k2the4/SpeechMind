"""Download the public HinEmo-9 repository data files into SpeechMind's ignored data folder.

The dataset's annotation/curation license is CC BY 4.0. The dataset authors
state that the underlying YouTube comment text remains subject to third-party
platform terms; do not redistribute the downloaded comment text.
"""
from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

REPOSITORY_ZIP = (
    "https://codeload.github.com/im-rakeshsahani/hinemo9/zip/refs/heads/main"
)
DESTINATION = Path("data/hinemo9")
REQUIRED = (
    "data/train.jsonl",
    "data/dev.jsonl",
    "data/test.jsonl",
    "data/gold.jsonl",
    "data/checksums.txt",
)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    print("[HinEmo-9] Downloading dataset repository archive...")
    request = urllib.request.Request(
        REPOSITORY_ZIP,
        headers={"User-Agent": "SpeechMind-dataset-setup/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
        root = next(
            (name.split("/", 1)[0] for name in names if name.endswith("README.md")),
            None,
        )
        if root is None:
            raise RuntimeError("The downloaded archive did not contain README.md.")

        missing = [
            relative for relative in REQUIRED
            if f"{root}/{relative}" not in names
        ]
        if missing:
            raise RuntimeError(
                "The dataset archive is missing expected files: "
                + ", ".join(missing)
            )

        for relative in REQUIRED + ("README.md", "LICENSE"):
            source = f"{root}/{relative}"
            if source not in names:
                if relative == "LICENSE":
                    continue
                raise RuntimeError(f"Dataset archive is missing {relative}.")
            target = DESTINATION / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(source) as input_file, target.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)

    print(f"[HinEmo-9] Dataset files installed in {DESTINATION.resolve()}")
    print("[HinEmo-9] Verify checksums and review the dataset card before training.")


if __name__ == "__main__":
    main()
