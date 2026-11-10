"""Download the native and romanized fastText weights for AI4Bharat IndicLID."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0"
TARGETS = {
    "ftn": f"{BASE}/indiclid-ftn.zip",
    "ftr": f"{BASE}/indiclid-ftr.zip",
}


def download_model(name: str, url: str, destination: Path) -> None:
    print(f"[IndicLID] Downloading {name} model...")
    request = urllib.request.Request(
        url, headers={"User-Agent": "SpeechMind-IndicLID-setup/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        content = response.read()

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        candidates = [
            item for item in archive.namelist()
            if item.endswith("model_baseline_roman.bin")
        ]
        if not candidates:
            raise RuntimeError(
                f"{name} archive did not contain model_baseline_roman.bin. "
                "Check the official AI4Bharat IndicLID release page."
            )
        # The release archive may contain additional research artefacts. Only
        # the public fastText inference model is extracted into SpeechMind.
        source = candidates[0]
        target = destination / "model_baseline_roman.bin"
        with archive.open(source) as infile, target.open("wb") as outfile:
            outfile.write(infile.read())
    print(f"[IndicLID] Installed: {destination / 'model_baseline_roman.bin'}")


def main() -> None:
    root = Path("checkpoints/indiclid")
    for name, url in TARGETS.items():
        download_model(name, url, root / name)
    print("[IndicLID] Both model weights installed.")


if __name__ == "__main__":
    main()
