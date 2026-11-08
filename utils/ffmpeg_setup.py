"""
Optional FFmpeg discovery for audio decoding on Windows.

FFMPEG_BIN may point to a bin directory or an ffmpeg executable. We also
recognize common WinGet package layouts without depending on a username.
"""
import os
import shutil
from pathlib import Path

_DLL_DIRECTORY_HANDLES = []


def _candidate_ffmpeg_dirs():
    configured = os.environ.get("FFMPEG_BIN")
    if configured:
        path = Path(os.path.expandvars(os.path.expanduser(configured))).resolve()
        yield path.parent if path.is_file() else path

    executable = shutil.which("ffmpeg")
    if executable:
        yield Path(executable).resolve().parent

    winget_packages = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if winget_packages.is_dir():
        for package in winget_packages.glob("*FFmpeg*/*/bin"):
            yield package


def configure_ffmpeg():
    for directory in _candidate_ffmpeg_dirs():
        if not directory.is_dir():
            continue
        directory = str(directory)
        current_path = os.environ.get("PATH", "")
        path_entries = current_path.split(os.pathsep) if current_path else []
        if directory not in path_entries:
            os.environ["PATH"] = directory + os.pathsep + current_path

        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            try:
                _DLL_DIRECTORY_HANDLES.append(add_dll_directory(directory))
            except OSError:
                pass
        return directory
    return None


FFMPEG_BIN = configure_ffmpeg()
