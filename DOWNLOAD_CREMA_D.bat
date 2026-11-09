@echo off
setlocal EnableExtensions
title SpeechMind - Download CREMA-D Audio Only
cd /d "%~dp0"

set "TARGET=%~dp0data\crema_d\AudioWAV"
set "TEMP_REPO=%TEMP%\SpeechMind-CREMA-D"
set "LOG=%~dp0crema_d_download.log"

echo ==========================================================
echo SPEECHMIND - CREMA-D AUDIO DATASET
echo ==========================================================
echo This downloads only the official AudioWAV folder.
echo It does not download the VideoFlash directory.
echo Expected audio files: approximately 7,442 WAV recordings.
echo Ensure at least 2 GB of free disk space.
echo.
echo IMPORTANT:
echo 1. Register your intended use using the CREMA-D access form.
echo 2. Review the ODbL and Database Contents License terms.
echo 3. Do not upload the audio dataset to GitHub.
echo.
echo Official repository:
echo https://github.com/CheyneyComputerScience/CREMA-D
echo.
if exist "%TARGET%\*.wav" (
    echo CREMA-D WAV files already exist in:
    echo %TARGET%
    echo Remove or rename that folder only if you intentionally need a fresh download.
    goto FINISH
)

where git.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not on PATH. Reopen VS Code or repair Git PATH first.
    goto FAILED
)

git lfs version >nul 2>&1
if errorlevel 1 (
    echo Git LFS is not installed.
    echo Install Git LFS from https://git-lfs.com/ and rerun this file.
    goto FAILED
)

git lfs install
if errorlevel 1 goto FAILED

if exist "%TEMP_REPO%" rmdir /s /q "%TEMP_REPO%"

echo.
echo Cloning repository metadata with LFS downloads disabled...
set "GIT_LFS_SKIP_SMUDGE=1"
git clone --filter=blob:none --no-checkout https://github.com/CheyneyComputerScience/CREMA-D.git "%TEMP_REPO%" > "%LOG%" 2>&1
if errorlevel 1 (
    echo Clone failed. Check crema_d_download.log.
    goto FAILED
)

pushd "%TEMP_REPO%"
git sparse-checkout init --cone >> "%LOG%" 2>&1
if errorlevel 1 goto CLONE_FAILED
git sparse-checkout set AudioWAV >> "%LOG%" 2>&1
if errorlevel 1 goto CLONE_FAILED
git checkout master >> "%LOG%" 2>&1
if errorlevel 1 goto CLONE_FAILED

echo.
echo Downloading only AudioWAV through Git LFS...
git lfs pull --include="AudioWAV/**" --exclude="" >> "%LOG%" 2>&1
if errorlevel 1 goto CLONE_FAILED

if not exist "AudioWAV" goto CLONE_FAILED
for /f %%N in ('dir /b /s "AudioWAV\*.wav" 2^>nul ^| find /c /v ""') do set "COUNT=%%N"
if not defined COUNT set "COUNT=0"
if "%COUNT%"=="0" goto CLONE_FAILED

popd
if not exist "%TARGET%" mkdir "%TARGET%"
robocopy "%TEMP_REPO%\AudioWAV" "%TARGET%" *.wav /E /R:2 /W:2
if errorlevel 8 goto FAILED

echo.
echo Downloaded WAV files: %COUNT%
echo Destination: %TARGET%
echo.
echo Next, run TRAIN_VOICE_EMOTION.bat to start the expanded experiment.
goto FINISH

:CLONE_FAILED
popd
echo Download failed during sparse checkout or Git LFS.
echo See crema_d_download.log.
goto FAILED

:FAILED
echo.
echo CREMA-D download did not complete.
echo Log: %LOG%
pause
exit /b 1

:FINISH
echo.
echo Log: %LOG%
pause
exit /b 0
