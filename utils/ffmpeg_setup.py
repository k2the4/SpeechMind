"""
Optional FFmpeg DLL discovery.

The application should work with FFmpeg on PATH. FFMPEG_BIN can also point
to an FFmpeg bin directory. This avoids a machine-specific absolute path.
"""
import os
import shutil

_DLL_DIRECTORY_HANDLES = []


def _candidate_ffmpeg_dirs():
    configured = os.environ.get("FFMPEG_BIN")
    if configured:
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(configured)))
        if os.path.isfile(path):
            path = os.path.dirname(path)
        yield path

    executable = shutil.which("ffmpeg")
    if executable:
        yield os.path.dirname(os.path.abspath(executable))


def configure_ffmpeg_dlls():
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return None

    for directory in _candidate_ffmpeg_dirs():
        if os.path.isdir(directory):
            try:
                handle = add_dll_directory(directory)
                _DLL_DIRECTORY_HANDLES.append(handle)
                return directory
            except OSError:
                continue
    return None


FFMPEG_BIN = configure_ffmpeg_dlls()
