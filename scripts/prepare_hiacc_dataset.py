"""Download and prepare the public HiACC Hinglish ASR corpus.

Zenodo record: 10.5281/zenodo.15551669
The archive is ~531.6 MB. Its MD5 is checked before extraction.
Only code and scripts are committed; the audio/transcripts stay under data/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "hiacc"
ARCHIVE_PATH = DATA_ROOT / "Corpus.zip"
EXTRACT_ROOT = DATA_ROOT / "raw"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
SUMMARY_PATH = DATA_ROOT / "dataset_summary.json"

DOWNLOAD_URL = "https://zenodo.org/records/15551669/files/Corpus.zip?download=1"
EXPECTED_MD5 = "dd6cc9354e1dee5e2f25bc5243df88ac"
CHUNK_SIZE = 8 * 1024 * 1024


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(path: Path = ARCHIVE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and md5_file(path).lower() == EXPECTED_MD5:
        print(f"[HiACC] Verified existing archive: {path}")
        return
    if path.exists():
        print("[HiACC] Existing archive checksum does not match; replacing it.")
        path.unlink()

    print("[HiACC] Downloading the 531.6 MB public Zenodo archive...")
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={"User-Agent": "SpeechMind research data preparation/1.0"},
    )
    downloaded = 0
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100.0 / total
                print(
                    f"\r[HiACC] {downloaded / (1024 * 1024):.1f} MB "
                    f"({pct:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\r[HiACC] {downloaded / (1024 * 1024):.1f} MB",
                    end="",
                    flush=True,
                )
    print()
    actual = md5_file(path)
    if actual.lower() != EXPECTED_MD5:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"HiACC archive checksum mismatch: expected {EXPECTED_MD5}, got {actual}. "
            "The incomplete/corrupt archive was deleted."
        )
    print("[HiACC] Archive checksum verified.")


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        bad_member = zf.testzip()
        if bad_member:
            raise RuntimeError(f"ZIP CRC check failed for {bad_member}")
        for member in zf.infolist():
            member_path = (destination / member.filename).resolve()
            if member_path != resolved_root and resolved_root not in member_path.parents:
                raise RuntimeError(f"Unsafe path in dataset archive: {member.filename}")
        zf.extractall(destination)


def _domain(parts: tuple[str, ...]) -> str:
    lowered = [part.lower() for part in parts]
    if "adult" in lowered:
        return "adult"
    if "children" in lowered or "child" in lowered:
        return "children"
    return "unknown"


def _split(parts: tuple[str, ...]) -> str:
    """Infer train/test/validation from common directory naming conventions."""
    aliases = {
        "train": "train", "training": "train", "trainset": "train",
        "trainingset": "train", "tr": "train", "trn": "train",
        "test": "test", "testing": "test", "testset": "test",
        "testingset": "test", "ts": "test", "tst": "test",
        "val": "val", "valid": "val", "validation": "val",
        "dev": "val", "devset": "val", "validset": "val",
        "validationset": "val",
    }
    for part in parts:
        name = Path(str(part)).stem.lower()
        compact = re.sub(r"[^a-z]", "", name)
        if compact in aliases:
            return aliases[compact]
        tokens = [token for token in re.split(r"[^a-z]+", name) if token]
        for token in tokens:
            if token in aliases:
                return aliases[token]
    return "unknown"


def _speaker_from_name(name: str) -> str:
    patterns = (
        r"(?:speaker|spk)[_-]?([a-z0-9]+)",
        r"\b([AC]\d{1,3})\b",
    )
    stem = Path(name).stem
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _parse_transcript_file(path: Path) -> list[tuple[str, str]]:
    """Read corpus CSV/TXT rows containing WAV filename, transcript."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    rows: list[tuple[str, str]] = []
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return rows
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    filename = next(
                        (str(item[k]) for k in item if "audio" in k.lower() or "file" in k.lower()),
                        "",
                    )
                    transcript = next(
                        (str(item[k]) for k in item if "text" in k.lower() or "transcript" in k.lower()),
                        "",
                    )
                    if filename and transcript:
                        rows.append((Path(filename).name, transcript.strip()))
        return rows

    lines = text.splitlines()
    for line in lines:
        if not line.strip():
            continue
        try:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            parsed = next(csv.reader([line], delimiter=delimiter))
        except (csv.Error, StopIteration, ValueError):
            parsed = [line]
        parsed = [cell.strip() for cell in parsed]
        filename_index = next(
            (
                i for i, cell in enumerate(parsed)
                if ".wav" in cell.lower()
            ),
            None,
        )
        if filename_index is None:
            # Support a mirrored one-transcript-per-file layout.
            if path.suffix.lower() == ".txt" and len(lines) == 1:
                rows.append((path.stem + ".wav", parsed[0]))
            continue
        candidate = parsed[filename_index]
        match = re.search(r"[^\\/\s,;]+\.wav", candidate, flags=re.IGNORECASE)
        if not match:
            continue
        filename = Path(match.group(0)).name
        transcript_parts = parsed[filename_index + 1 :]
        transcript = ",".join(part for part in transcript_parts if part).strip()
        if not transcript and filename_index > 0:
            transcript = ",".join(parsed[:filename_index]).strip()
        if filename.lower() in {"audio.wav", "filename.wav", "file.wav"}:
            continue
        if transcript:
            rows.append((filename, transcript))
    return rows



def _speaker_metadata_map(extract_root: Path) -> dict[str, str]:
    """Best-effort filename -> speaker ID mapping from sentence_stats metadata."""
    mapping: dict[str, str] = {}
    for metadata_path in extract_root.rglob("sentence_stats.csv"):
        try:
            with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    normalized = {
                        str(key or "").strip().lower(): str(value or "").strip()
                        for key, value in row.items()
                    }
                    speaker_id = next(
                        (
                            value for key, value in normalized.items()
                            if "speaker" in key and value
                        ),
                        "",
                    )
                    audio_value = next(
                        (
                            value for value in normalized.values()
                            if ".wav" in value.lower()
                        ),
                        "",
                    )
                    if speaker_id and audio_value:
                        match = re.search(
                            r"[^\\/\s,;]+\.wav",
                            audio_value,
                            flags=re.IGNORECASE,
                        )
                        if match:
                            mapping[Path(match.group(0)).name.lower()] = speaker_id
        except (OSError, csv.Error):
            continue
    return mapping


def build_manifest(extract_root: Path = EXTRACT_ROOT) -> list[dict]:
    audio_files = [
        path for path in extract_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
        and "audio" in [part.lower() for part in path.parts]
    ]
    if not audio_files:
        raise RuntimeError(
            f"No WAV audio files found under {extract_root}. "
            "Inspect dataset_summary.json and update the parser for the archive layout."
        )

    audio_lookup: dict[tuple[str, str, str], Path] = {}
    for audio_path in audio_files:
        relative = audio_path.relative_to(extract_root)
        parts = relative.parts
        domain = _domain(parts)
        split = _split(parts)
        audio_lookup[(domain, split, audio_path.name.lower())] = audio_path

    speaker_metadata = _speaker_metadata_map(extract_root)
    # Transcript files in the archive may be named .txt/.csv without a
    # parent folder literally called "transcripts". Search small text files
    # across the corpus, excluding metadata and annotation tables; the parser
    # accepts a file only when rows actually contain a WAV filename.
    transcript_paths = [
        path for path in extract_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".txt", ".tsv", ".json"}
        and not any(
            part.lower() in {"metadata", "annotations", "annotation"}
            for part in path.parts
        )
        and path.stat().st_size <= 50 * 1024 * 1024
    ]


    rows: list[dict] = []
    matched_audio: set[Path] = set()
    unmatched_transcripts = 0
    for transcript_path in transcript_paths:
        relative = transcript_path.relative_to(extract_root)
        domain = _domain(relative.parts)
        split = _split(relative.parts)
        for filename, transcript in _parse_transcript_file(transcript_path):
            key = (domain, split, filename.lower())
            audio_path = audio_lookup.get(key)
            if audio_path is None:
                # Transcript indexes may be outside the split tree. Match
                # by basename first, then use known domain/split labels to
                # disambiguate only when they produce candidates.
                matches = [
                    value for (d, split_value, name), value in audio_lookup.items()
                    if name == filename.lower()
                ]
                if domain != "unknown":
                    narrowed = [
                        value for value in matches
                        if _domain(value.relative_to(extract_root).parts) == domain
                    ]
                    if narrowed:
                        matches = narrowed
                if split != "unknown":
                    narrowed = [
                        value for value in matches
                        if _split(value.relative_to(extract_root).parts) == split
                    ]
                    if narrowed:
                        matches = narrowed
                if len(matches) == 1:
                    audio_path = matches[0]


            if audio_path is None:
                unmatched_transcripts += 1
                continue
            matched_audio.add(audio_path)
            audio_relative = audio_path.relative_to(extract_root)
            age_group = _domain(audio_relative.parts)
            audio_split = _split(audio_relative.parts)
            rows.append({
                "audio_path": str(audio_path.relative_to(DATA_ROOT)).replace("\\", "/"),
                "transcript": transcript,
                "age_group": age_group,
                "split": audio_split,
                "speaker_id": (
                    speaker_metadata.get(audio_path.name.lower())
                    or _speaker_from_name(audio_path.name)
                ),
                "audio_filename": audio_path.name,
                "transcript_source": str(transcript_path.relative_to(extract_root)).replace("\\", "/"),
            })

    # Remove duplicate entries from duplicate transcript indexes, preserving
    # the first canonical transcript encountered.
    unique: dict[tuple[str, str], dict] = {}
    for row in rows:
        unique.setdefault((row["audio_path"], row["transcript"]), row)
    rows = list(unique.values())
    if not rows:
        raise RuntimeError(
            "The HiACC audio files were extracted, but transcript rows could not "
            "be paired automatically. Do not delete the archive; send "
            "dataset_summary.json so the parser can be adjusted to the actual schema."
        )

    rows.sort(key=lambda row: (row["split"], row["age_group"], row["audio_filename"]))
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    split_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        group_key = f"{row['age_group']}/{row['split']}"
        group_counts[group_key] = group_counts.get(group_key, 0) + 1

    summary = {
        "dataset": "HiACC: Hinglish Adult & Children Code-switched corpus",
        "zenodo_record": "https://zenodo.org/records/15551669",
        "license_on_zenodo_record": "CC-BY-4.0",
        "archive": str(ARCHIVE_PATH),
        "audio_files_found": len(audio_files),
        "transcript_files_found": len(transcript_paths),
        "paired_manifest_rows": len(rows),
        "unmatched_transcript_rows": unmatched_transcripts,
        "audio_files_without_transcripts": len(audio_files) - len(matched_audio),
        "rows_by_split": split_counts,
        "rows_by_age_group_and_split": group_counts,
        "manifest": str(MANIFEST_PATH),
        "note": (
            "ASR benchmark corpus only; no emotion labels. Follow dataset citation "
            "and CC BY 4.0 attribution requirements."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[HiACC] Dataset prepared.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return rows


def main() -> int:
    try:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        download_archive()
        has_audio = (
            EXTRACT_ROOT.exists()
            and any(EXTRACT_ROOT.rglob("*.wav"))
        )
        if not has_audio:
            if EXTRACT_ROOT.exists():
                shutil.rmtree(EXTRACT_ROOT)
            print("[HiACC] Extracting corpus archive...")
            safe_extract(ARCHIVE_PATH, EXTRACT_ROOT)
        rows = build_manifest()
        if not any(row["split"] == "test" for row in rows):
            raise RuntimeError(
                "The manifest has no test split. The archive is already extracted. "
                "Run INSPECT_HIACC_LAYOUT.bat to inspect its actual paths; do not "
                "download or extract it again."
            )
        (DATA_ROOT / "ATTRIBUTION.txt").write_text(
            "HiACC: Hinglish Adult & Children Code-switched corpus\n"
            "Creators: Shruti Singh, Muskaan Singh, Virender Kadyan\n"
            "Zenodo: https://zenodo.org/records/15551669\n"
            "DOI: 10.5281/zenodo.15551669\n"
            "License listed by the Zenodo record: Creative Commons Attribution 4.0 International.\n"
            "Please retain attribution and cite the dataset when using it.\n",
            encoding="utf-8",
        )
        print(f"\n[HiACC] Manifest ready: {MANIFEST_PATH}")
        return 0
    except Exception as error:
        print(f"\n[HiACC] ERROR: {type(error).__name__}: {error}")
        print("The raw archive and extracted files have been kept for diagnosis.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
