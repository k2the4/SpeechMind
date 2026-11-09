@echo off
setlocal EnableExtensions
title SpeechMind - Second PC Setup
cd /d "%~dp0"

set "LOG=%~dp0second_pc_setup.log"
set "PYTHON_EXE="
> "%LOG%" echo SpeechMind second-PC setup
>> "%LOG%" echo Started: %DATE% %TIME%
>> "%LOG%" echo Project: %CD%

echo ==================================================
echo SpeechMind - Second Windows PC setup
echo ==================================================
echo This installs the project into an isolated .venv.
echo It does not change your system Python installation.
echo.

if not exist "%~dp0requirements.txt" (
    echo ERROR: requirements.txt not found.
    echo Run this file from the extracted SpeechMind project folder.
    set "ERR=1"
    goto FINISH
)

where py >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python Launcher (py.exe) was not found.
    echo Install Python 3.11 from python.org and enable the Python Launcher.
    set "ERR=2"
    goto FINISH
)

py -3.11 --version >> "%LOG%" 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('py -3.11 -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%P"
)

if not defined PYTHON_EXE (
    py -3.10 --version >> "%LOG%" 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%P in ('py -3.10 -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%P"
    )
)

if not defined PYTHON_EXE (
    echo ERROR: Python 3.11 or 3.10 was not found.
    echo Python 3.14 is not supported for this project setup.
    echo Install Python 3.11, then run this file again.
    set "ERR=3"
    goto FINISH
)

echo Using:
echo %PYTHON_EXE%
>> "%LOG%" echo Using Python: %PYTHON_EXE%

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo.
    echo Creating isolated project environment...
    "%PYTHON_EXE%" -m venv "%~dp0.venv" >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo ERROR: Could not create .venv. See second_pc_setup.log.
        set "ERR=4"
        goto FINISH
    )
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
echo.
echo Updating pip...
"%VENV_PY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: pip update failed. See second_pc_setup.log.
    set "ERR=5"
    goto FINISH
)

echo.
echo Installing SpeechMind dependencies.
echo This may take several minutes and download large packages.
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    echo See second_pc_setup.log for the exact package error.
    set "ERR=6"
    goto FINISH
)

echo.
echo Checking core imports, PyTorch and GPU...
"%VENV_PY%" -c "import sys,torch,torchaudio,transformers,streamlit,librosa,soundfile,sklearn; print('Python:',sys.version); print('PyTorch:',torch.__version__); print('TorchAudio:',torchaudio.__version__); print('Transformers:',transformers.__version__); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU detected - CPU mode'); print('SpeechMind imports: OK')" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: A core import or PyTorch check failed.
    echo See second_pc_setup.log.
    set "ERR=7"
    goto FINISH
)

echo.
echo Running project unit tests...
"%VENV_PY%" -m unittest discover -s "%~dp0tests" -v >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: Unit tests failed. See second_pc_setup.log.
    set "ERR=8"
    goto FINISH
)

if not exist "%~dp0data\ravdess" mkdir "%~dp0data\ravdess"
if not exist "%~dp0training_outputs\wavlm_ravdess" mkdir "%~dp0training_outputs\wavlm_ravdess"

echo.
echo Checking FFmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo FFmpeg was not found on PATH.
    echo Whisper audio decoding may require FFmpeg.
    where winget >nul 2>&1
    if not errorlevel 1 (
        choice /C YN /N /M "Install FFmpeg using Windows Package Manager now? [Y/N] "
        if errorlevel 2 goto FFMPEG_SKIP
        winget install --id Gyan.FFmpeg.Shared -e --accept-package-agreements --accept-source-agreements >> "%LOG%" 2>&1
        if errorlevel 1 (
            echo FFmpeg installation did not complete. See setup log.
        ) else (
            echo FFmpeg installation command completed.
            echo Open a NEW terminal after setup so PATH refreshes.
        )
    ) else (
        echo Windows Package Manager was not found; FFmpeg was not installed.
    )
) else (
    ffmpeg -version >> "%LOG%" 2>&1
    echo FFmpeg is available.
)

:FFMPEG_SKIP
echo.
echo ==================================================
echo SETUP CHECKS PASSED
echo ==================================================
echo Python environment: .venv
echo Dataset location expected: data\ravdess
echo Training launcher: TRAIN_VOICE_EMOTION.bat
echo Setup log: second_pc_setup.log
echo.
echo IMPORTANT: Copy the RAVDESS actor folders into data\ravdess.
echo The dataset is not stored in GitHub and is not downloaded by setup.
echo Do not start training until the dataset is present.
set "ERR=0"
goto FINISH

:FINISH
if not "%ERR%"=="0" (
    echo.
    echo SETUP DID NOT COMPLETE. Error code: %ERR%
    echo Send second_pc_setup.log for diagnosis.
)
echo.
echo Log: %LOG%
echo Press any key to close this window.
pause >nul
exit /b %ERR%
