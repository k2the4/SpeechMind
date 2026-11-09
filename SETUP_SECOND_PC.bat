@echo off
setlocal EnableExtensions
title SpeechMind - Complete Windows Setup
cd /d "%~dp0"

set "LOG=%~dp0second_pc_setup.log"
set "ERR=0"
set "PYTHON_EXE="
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

> "%LOG%" echo SpeechMind complete Windows setup
>> "%LOG%" echo Started: %DATE% %TIME%
>> "%LOG%" echo Project: %CD%

echo ==========================================================
echo SPEECHMIND - COMPLETE WINDOWS SETUP
echo ==========================================================
echo This installer prepares Python, FFmpeg, PyTorch, packages,
echo the project environment, and runs the automated tests.
echo It does not upload or delete your datasets.
echo.

if not exist "%~dp0requirements.txt" (
    echo ERROR: Project files are missing.
    echo First clone https://github.com/k2the4/SpeechMind.git into this folder.
    set "ERR=1"
    goto FINISH
)

where winget >nul 2>&1
if errorlevel 1 (
    echo ERROR: Windows Package Manager (winget) is unavailable.
    echo Install or update App Installer from Microsoft Store, then retry.
    set "ERR=2"
    goto FINISH
)

echo [1/7] Checking Python 3.11...
py -3.11 --version >> "%LOG%" 2>&1
if errorlevel 1 (
    echo Python 3.11 was not found. Installing it now...
    winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo ERROR: Python 3.11 installation failed. See setup log.
        set "ERR=3"
        goto FINISH
    )
)

for /f "delims=" %%P in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYTHON_EXE if exist "%ProgramFiles%\Python311\python.exe" set "PYTHON_EXE=%ProgramFiles%\Python311\python.exe"
if not defined PYTHON_EXE (
    echo Python was installed but cannot be located in this terminal.
    echo Close VS Code, reopen it, and double-click this setup file again.
    set "ERR=4"
    goto FINISH
)
echo Using Python:
echo %PYTHON_EXE%
>> "%LOG%" echo Python executable: %PYTHON_EXE%
"%PYTHON_EXE%" --version >> "%LOG%" 2>&1

echo.
echo [2/7] Checking FFmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo Installing FFmpeg...
    winget install --id Gyan.FFmpeg -e --silent --accept-package-agreements --accept-source-agreements >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo WARNING: FFmpeg installation did not complete. See log.
        echo Continuing; audio decoding may need FFmpeg PATH setup later.
        >> "%LOG%" echo WARNING: FFmpeg installation failed.
    )
) else (
    echo FFmpeg already available.
)
if exist "%~dp0utils\ffmpeg_setup.py" (
    echo Project FFmpeg auto-discovery is enabled.
)

echo.
echo [3/7] Creating the isolated project environment...
if not exist "%VENV_PY%" (
    "%PYTHON_EXE%" -m venv "%~dp0.venv" >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo ERROR: Could not create .venv. See setup log.
        set "ERR=5"
        goto FINISH
    )
) else (
    echo Existing .venv found; keeping it.
)

echo.
echo [4/7] Updating pip...
"%VENV_PY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: pip update failed. See setup log.
    set "ERR=6"
    goto FINISH
)

echo.
echo [5/7] Installing PyTorch for this computer...
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo NVIDIA driver detected. Installing PyTorch CUDA 12.8 wheels...
    nvidia-smi >> "%LOG%" 2>&1
    "%VENV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 >> "%LOG%" 2>&1
) else (
    echo No NVIDIA driver detected. Installing the PyTorch CPU build...
    "%VENV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu >> "%LOG%" 2>&1
)
if errorlevel 1 (
    echo ERROR: PyTorch installation failed. See setup log.
    set "ERR=7"
    goto FINISH
)

echo.
echo [6/7] Installing SpeechMind packages...
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: Dependency installation failed. See setup log.
    set "ERR=8"
    goto FINISH
)

echo.
echo Checking imports, CUDA, and audio tools...
"%VENV_PY%" -c "import sys,torch,torchaudio,transformers,streamlit,librosa,soundfile,sklearn; print('Python:',sys.version); print('PyTorch:',torch.__version__); print('TorchAudio:',torchaudio.__version__); print('Transformers:',transformers.__version__); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode'); print('Core imports: OK')" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: Core imports failed. See setup log.
    set "ERR=9"
    goto FINISH
)

echo.
echo [7/7] Running automated project tests...
"%VENV_PY%" -m unittest discover -s "%~dp0tests" -v >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: Unit tests failed. See setup log.
    set "ERR=10"
    goto FINISH
)

if not exist "%~dp0data\ravdess" mkdir "%~dp0data\ravdess"
if not exist "%~dp0checkpoints" mkdir "%~dp0checkpoints"
if not exist "%~dp0training_outputs\wavlm_ravdess" mkdir "%~dp0training_outputs\wavlm_ravdess"

echo.
echo ==========================================================
echo SETUP COMPLETE
echo ==========================================================
echo Python environment: .venv
echo PyTorch and SpeechMind dependencies: installed
echo Unit tests: passed
echo Dataset folder: data\ravdess
echo Training launcher: TRAIN_VOICE_EMOTION.bat
echo.
echo IMPORTANT: Copy the RAVDESS Actor_01 through Actor_24 folders
echo into data\ravdess before training. Audio data is not in GitHub.
echo The WavLM model weights download automatically when training starts.
goto FINISH

:FINISH
if not "%ERR%"=="0" (
    echo.
    echo SETUP DID NOT COMPLETE. Error code: %ERR%
    echo Please share second_pc_setup.log for diagnosis.
)
echo.
echo Full setup log: %LOG%
echo Press any key to close this window.
pause >nul
exit /b %ERR%
