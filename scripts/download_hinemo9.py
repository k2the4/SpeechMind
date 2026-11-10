"""Download and checksum-verify the public HinEmo-9 data files.

The dataset authors' CC BY 4.0 license covers annotations/curation, not
underlying third-party YouTube comment text. Keep the data local; do not
redistribute comment text.
"""
from __future__ import annotations

import hashlib
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(destination: Path) -> None:
    checksum_file = destination / "data" / "checksums.txt"
    expected = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            expected[parts[1].lstrip("*")] = parts[0].lower()

    required_data = ("train.jsonl", "dev.jsonl", "test.jsonl", "gold.jsonl")
    for filename in required_data:
        expected_hash = expected.get(filename)
        if not expected_hash:
            raise RuntimeError(f"No SHA-256 checksum found for {filename}.")
        actual_hash = sha256(destination / "data" / filename)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )
    print("[HinEmo-9] SHA-256 checks passed for all four dataset splits.")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    print("[HinEmo-9] Downloading dataset repository archive...")
    request = urllib.request.Request(
        REPOSITORY_ZIP,
        headers={"User-Agent": "SpeechMind-dataset-setup/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
        root_entry = next(
            (name for name in names if name.endswith("/data/train.jsonl")),
            None,
        )
        if root_entry is None:
            raise RuntimeError("The downloaded archive did not contain data/train.jsonl.")
        root = root_entry.split("/", 1)[0]

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

    verify_checksums(DESTINATION)
    print(f"[HinEmo-9] Dataset files installed in {DESTINATION.resolve()}")
    print("[HinEmo-9] Review README.md and docs/dataset_card.md before training.")


if __name__ == "__main__":
    main()
