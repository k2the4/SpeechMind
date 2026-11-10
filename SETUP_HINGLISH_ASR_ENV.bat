@echo off
setlocal
cd /d "%~dp0"

set "VENV=%~dp0.venv_srota"
set "PYTHON_EXE=%VENV%\Scripts\python.exe"

echo ==========================================================
echo SpeechMind - Isolated Srota Hinglish ASR environment
echo ==========================================================
echo This does NOT modify the existing .venv.
echo It installs CPU PyTorch and qwen-asr into .venv_srota.
echo The optional model is approximately 0.8B parameters.
echo CPU inference may be slow; the first model run downloads weights.
echo.

if not exist "%VENV%\Scripts\python.exe" (
    py -3.11 -m venv "%VENV%"
    if errorlevel 1 goto failed
)

"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto failed

echo Installing CPU-only PyTorch...
"%PYTHON_EXE%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed

echo Installing Srota inference package...
"%PYTHON_EXE%" -m pip install -r requirements_hinglish_asr.txt
if errorlevel 1 goto failed

echo Verifying imports...
"%PYTHON_EXE%" -c "import torch, transformers; from qwen_asr import Qwen3ASRModel; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('Qwen3ASRModel: OK'); print('CUDA:', torch.cuda.is_available())"
if errorlevel 1 goto failed

echo.
echo Srota environment is ready at .venv_srota.
echo Existing SpeechMind .venv was not changed.
pause
exit /b 0

:failed
echo.
echo Srota setup failed. Existing SpeechMind .venv is unchanged.
echo Review the error above; do not reinstall SpeechMind's main dependencies.
pause
exit /b 1
