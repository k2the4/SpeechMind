"""Write a concise inventory of an already-extracted HiACC archive.

This does not download, extract, move, or modify dataset files. It is used to
inspect the real corpus layout when transcript pairing/split detection fails.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "hiacc"
RAW_ROOT = DATA_ROOT / "raw"
REPORT = DATA_ROOT / "hiacc_layout_report.txt"
TEXT_EXTENSIONS = {".csv", ".txt", ".tsv", ".json"}


def domain(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "adult" in parts:
        return "adult"
    if "children" in parts or "child" in parts:
        return "children"
    return "unknown"


def split_name(path: Path) -> str:
    aliases = {
        "train": "train", "training": "train", "trainset": "train",
        "test": "test", "testing": "test", "testset": "test",
        "val": "val", "valid": "val", "validation": "val", "dev": "val",
    }
    for part in path.parts:
        name = Path(part).stem.lower()
        compact = re.sub(r"[^a-z]", "", name)
        if compact in aliases:
            return aliases[compact]
        for token in re.split(r"[^a-z]+", name):
            if token in aliases:
                return aliases[token]
    return "unknown"


def preview(path: Path, max_lines: int = 4) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError as error:
        return [f"<read error: {error}>"]
    return [
        line[:500] + ("..." if len(line) > 500 else "")
        for line in lines[:max_lines]
    ]


def main() -> int:
    if not RAW_ROOT.is_dir():
        raise SystemExit(
            f"Extracted HiACC folder not found: {RAW_ROOT}. "
            "Do not redownload; first verify the archive extraction path."
        )

    all_files = [p for p in RAW_ROOT.rglob("*") if p.is_file()]
    audio_files = [
        p for p in all_files
        if p.suffix.lower() == ".wav"
        and any(part.lower() == "audio" for part in p.parts)
    ]
    text_files = [
        p for p in all_files
        if p.suffix.lower() in TEXT_EXTENSIONS
        and not any(
            part.lower() in {"metadata", "annotations", "annotation"}
            for part in p.parts
        )
        and p.stat().st_size <= 50 * 1024 * 1024
    ]
    metadata_files = [
        p for p in all_files
        if p.name.lower() in {"sentence_stats.csv", "speaker_info.csv"}
    ]

    lines = [
        "HIACC EXTRACTED ARCHIVE INVENTORY",
        f"Extract root: {RAW_ROOT}",
        f"Total files: {len(all_files)}",
        f"WAV audio files: {len(audio_files)}",
        f"Potential transcript/index files: {len(text_files)}",
        f"Metadata CSV files: {len(metadata_files)}",
        "",
        "AUDIO COUNTS BY DOMAIN AND DETECTED SPLIT",
    ]
    counts = Counter((domain(p.relative_to(RAW_ROOT)), split_name(p.relative_to(RAW_ROOT))) for p in audio_files)
    for key, number in sorted(counts.items()):
        lines.append(f"{key[0]}/{key[1]}: {number}")

    lines.extend(["", "AUDIO DIRECTORY SAMPLES"])
    for key in sorted(counts):
        samples = [
            p for p in audio_files
            if (domain(p.relative_to(RAW_ROOT)), split_name(p.relative_to(RAW_ROOT))) == key
        ][:5]
        lines.append(f"[{key[0]}/{key[1]}]")
        lines.extend(f"  {p.relative_to(RAW_ROOT)}" for p in samples)

    lines.extend(["", "POTENTIAL TRANSCRIPT FILES AND FIRST LINES"])
    wav_pattern = re.compile(r"[^\\/\s,;]+\.wav", re.IGNORECASE)
    for path in sorted(text_files):
        rel = path.relative_to(RAW_ROOT)
        sample = preview(path)
        full_preview = preview(path, max_lines=1000)
        wav_refs = sum(len(wav_pattern.findall(line)) for line in full_preview)
        lines.append(
            f"[{rel}] bytes={path.stat().st_size} "
            f"domain={domain(rel)} split={split_name(rel)} "
            f"lines_previewed={len(full_preview)} wav_references={wav_refs}"
        )
        lines.extend(f"  {line}" for line in sample)

    lines.extend(["", "METADATA CSV HEADERS AND SAMPLE ROWS"])
    for path in sorted(metadata_files):
        lines.append(f"[{path.relative_to(RAW_ROOT)}]")
        lines.extend(f"  {line}" for line in preview(path, max_lines=4))

    lines.extend([
        "",
        "NOTE: This inventory reads only text metadata/transcript files and paths.",
        "It does not open, transcribe, modify, or delete audio.",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[HiACC] Full report saved to: {REPORT}")
    return 0


if __name__ == "__main__":
    main()
