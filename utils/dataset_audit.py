"""
Local dataset audit for SpeechMind.
Discovers the project's Test Data folder, counts audio/metadata files, and parses RAVDESS speech filenames.
It never modifies or uploads recordings.
"""
from collections import Counter
from pathlib import Path

RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
RAVDESS_STATEMENTS = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}


def discover_test_data_roots(home=None, cwd=None):
    home = Path(home or Path.home())
    cwd = Path(cwd or Path.cwd())
    candidates = [
        home / "OneDrive" / "Desktop" / "Test Data",
        home / "Desktop" / "Test Data",
        cwd / "Test Data",
        cwd.parent / "Test Data",
    ]
    roots, seen = [], set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        key = str(resolved).lower()
        if key not in seen and resolved.is_dir():
            seen.add(key)
            roots.append(resolved)
    return roots


def parse_ravdess_filename(filename):
    path = Path(filename)
    if path.suffix.lower() != ".wav":
        return None
    parts = path.stem.split("-")
    if len(parts) != 7 or not all(part.isdigit() for part in parts):
        return None
    modality, channel, emotion, intensity, statement, repetition, actor = parts
    if modality != "03" or channel != "01" or emotion not in RAVDESS_EMOTIONS:
        return None
    if statement not in RAVDESS_STATEMENTS:
        return None
    return {
        "emotion": RAVDESS_EMOTIONS[emotion],
        "emotion_code": emotion,
        "actor": actor,
        "statement": RAVDESS_STATEMENTS[statement],
        "intensity": intensity,
        "repetition": repetition,
    }


def audit_local_datasets(roots=None):
    roots = roots if roots is not None else discover_test_data_roots()
    report = {
        "roots": [str(root) for root in roots],
        "root_count": len(roots),
        "total_files": 0,
        "wav_files": 0,
        "csv_files": 0,
        "xlsx_files": 0,
        "other_audio_files": 0,
        "ravdess_speech_wavs": 0,
        "ravdess_emotion_counts": {},
        "ravdess_actor_count": 0,
        "ravdess_actors": [],
        "hindi_folder_wavs": 0,
        "hindi_folder_found": False,
        "other_wav_examples": [],
        "warnings": [],
    }
    emotion_counts, actors, other_audio = Counter(), set(), []
    all_files = []
    for root in roots:
        try:
            all_files.extend(p for p in root.rglob("*") if p.is_file())
        except OSError as error:
            report["warnings"].append(f"Could not fully scan {root}: {error}")

    report["total_files"] = len(all_files)
    report["wav_files"] = sum(p.suffix.lower() == ".wav" for p in all_files)
    report["csv_files"] = sum(p.suffix.lower() == ".csv" for p in all_files)
    report["xlsx_files"] = sum(p.suffix.lower() in {".xlsx", ".xls"} for p in all_files)
    report["other_audio_files"] = sum(
        p.suffix.lower() in {".mp3", ".flac", ".m4a", ".ogg"} for p in all_files
    )

    for path in all_files:
        parsed = parse_ravdess_filename(path.name)
        if parsed:
            report["ravdess_speech_wavs"] += 1
            emotion_counts[parsed["emotion"]] += 1
            actors.add(parsed["actor"])
        elif path.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
            if len(other_audio) < 10:
                other_audio.append(path.name)
        if path.parent.name.lower() == "hindi" and path.suffix.lower() == ".wav":
            report["hindi_folder_wavs"] += 1
            report["hindi_folder_found"] = True

    report["ravdess_emotion_counts"] = dict(sorted(emotion_counts.items()))
    report["ravdess_actor_count"] = len(actors)
    report["ravdess_actors"] = sorted(actors)
    report["other_wav_examples"] = other_audio

    if not roots:
        report["warnings"].append("No Test Data directory was found at the known project locations.")
    if report["ravdess_speech_wavs"]:
        if report["ravdess_speech_wavs"] != 1440:
            report["warnings"].append(
                "RAVDESS speech file count differs from the expected 1,440; the folder may be partial."
            )
        if report["ravdess_actor_count"] < 24:
            report["warnings"].append(
                "Fewer than 24 RAVDESS actors were found; check whether the download is complete."
            )
        report["warnings"].append(
            "The current Dpngtm voice-emotion checkpoint lists RAVDESS among its training "
            "sources. Do not use this folder as an independent voice-emotion test set."
        )
    if report["hindi_folder_wavs"]:
        report["warnings"].append(
            "Hindi WAV files are present, but audio alone is not an emotion label. "
            "A labelled Hindi speech-emotion corpus is still needed for SER training/evaluation."
        )
    if report["wav_files"] and not report["ravdess_speech_wavs"]:
        report["warnings"].append(
            "WAV files were found, but their filenames do not match the RAVDESS speech convention."
        )
    return report
