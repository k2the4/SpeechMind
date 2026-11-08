import os

FFMPEG_BIN = r"C:\Users\kvasi\AppData\Local\Microsoft\WinGet\Packages\BtbN.FFmpeg.GPL.Shared.7.1_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-n7.1.5-12-g1fdbca85aa-win64-gpl-shared-7.1\bin"

if os.path.isdir(FFMPEG_BIN):
    os.add_dll_directory(FFMPEG_BIN)