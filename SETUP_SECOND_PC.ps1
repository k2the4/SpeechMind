# SpeechMind complete Windows setup
# Runs from the repository root. All dependencies are installed into .venv.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$LogPath = Join-Path $ProjectRoot "second_pc_setup.log"

try {
    Start-Transcript -Path $LogPath -Force | Out-Null
    Write-Host "=========================================================="
    Write-Host "SPEECHMIND - COMPLETE WINDOWS SETUP"
    Write-Host "=========================================================="
    Write-Host "Project: $ProjectRoot"
    Write-Host "This prepares Python, FFmpeg, PyTorch and project packages."
    Write-Host "It does not upload, delete, or move your datasets."
    Write-Host ""

    if (-not (Test-Path (Join-Path $ProjectRoot "requirements.txt"))) {
        throw "requirements.txt is missing. Clone SpeechMind into this folder first."
    }

    $Winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $Winget) {
        throw "Windows Package Manager (winget) is unavailable. Install/update App Installer from Microsoft Store."
    }

    # 1. Python 3.11
    Write-Host "[1/7] Checking Python 3.11..."
    $PythonExe = $null
    $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        & py -3.11 --version
        if ($LASTEXITCODE -eq 0) {
            $PythonExe = (& py -3.11 -c "import sys; print(sys.executable)" | Select-Object -Last 1).Trim()
        }
    }

    if (-not $PythonExe) {
        Write-Host "Python 3.11 not found. Installing with WinGet..."
        & winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.11 installation failed. See the log above."
        }

        # Refresh PATH for this PowerShell process after the installer updates it.
        $MachinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
        $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $env:Path = "$MachinePath;$UserPath"

        $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($PyLauncher) {
            $PythonExe = (& py -3.11 -c "import sys; print(sys.executable)" | Select-Object -Last 1).Trim()
        }
        if (-not $PythonExe) {
            $Candidates = @(
                (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
                (Join-Path $env:ProgramFiles "Python311\python.exe"),
                (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages")
            )
            foreach ($Candidate in $Candidates) {
                if (Test-Path $Candidate -PathType Leaf) {
                    $PythonExe = $Candidate
                    break
                }
            }
        }
    }

    if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
        throw "Python 3.11 was installed but python.exe could not be located. Restart Windows/VS Code and run setup again."
    }
    & $PythonExe --version
    if ($LASTEXITCODE -ne 0) { throw "Python failed to start." }

    # 2. FFmpeg
    Write-Host ""
    Write-Host "[2/7] Checking FFmpeg..."
    $Ffmpeg = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
    if (-not $Ffmpeg) {
        Write-Host "Installing FFmpeg with WinGet..."
        & winget install --id Gyan.FFmpeg.Shared -e --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "FFmpeg installation did not complete. The project has an FFmpeg path discovery helper; audio decoding may need manual FFmpeg setup."
        }
        $MachinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
        $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $env:Path = "$MachinePath;$UserPath"
        $Ffmpeg = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
    }
    if ($Ffmpeg) {
        & ffmpeg -version | Select-Object -First 1
    } else {
        Write-Warning "FFmpeg is not visible on PATH in this session."
    }

    # 3. Local virtual environment
    Write-Host ""
    Write-Host "[3/7] Creating/checking project virtual environment..."
    $VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        & $PythonExe -m venv (Join-Path $ProjectRoot ".venv")
        if ($LASTEXITCODE -ne 0) { throw "Could not create .venv." }
    } else {
        Write-Host "Existing .venv found; keeping it."
    }

    if (-not (Test-Path $VenvPython)) { throw ".venv Python was not created." }

    # 4. pip
    Write-Host ""
    Write-Host "[4/7] Updating pip..."
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip update failed." }

    # 5. PyTorch build appropriate for the PC
    Write-Host ""
    Write-Host "[5/7] Installing PyTorch and TorchAudio..."
    $NvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($NvidiaSmi) {
        Write-Host "NVIDIA driver detected. Installing PyTorch CUDA 12.8 wheels."
        & nvidia-smi
        & $VenvPython -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    } else {
        Write-Host "No NVIDIA driver detected. Installing CPU-only PyTorch."
        & $VenvPython -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    }
    if ($LASTEXITCODE -ne 0) { throw "PyTorch/TorchAudio installation failed." }

    # 6. Project dependencies
    Write-Host ""
    Write-Host "[6/7] Installing SpeechMind packages..."
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "SpeechMind dependency installation failed." }

    # 7. Verify imports, CUDA and tests
    Write-Host ""
    Write-Host "[7/7] Verifying imports and GPU..."
    $VerifyCode = "import sys, torch, torchaudio, transformers, streamlit, librosa, soundfile, sklearn; print('Python:', sys.version); print('PyTorch:', torch.__version__); print('TorchAudio:', torchaudio.__version__); print('Transformers:', transformers.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode'); print('Core imports: OK')"
    & $VenvPython -c $VerifyCode
    if ($LASTEXITCODE -ne 0) { throw "Core Python imports or PyTorch verification failed." }

    Write-Host ""
    Write-Host "Running automated unit tests..."
    & $VenvPython -m unittest discover -s (Join-Path $ProjectRoot "tests") -v
    if ($LASTEXITCODE -ne 0) { throw "Project unit tests failed." }

    # Create local working folders. No dataset is downloaded or moved.
    @("data\ravdess", "checkpoints", "training_outputs\wavlm_ravdess") | ForEach-Object {
        $Directory = Join-Path $ProjectRoot $_
        if (-not (Test-Path $Directory)) {
            New-Item -ItemType Directory -Path $Directory -Force | Out-Null
        }
    }

    Write-Host ""
    Write-Host "=========================================================="
    Write-Host "SPEECHMIND SETUP COMPLETE"
    Write-Host "=========================================================="
    Write-Host "Python environment: .venv"
    Write-Host "PyTorch and project dependencies: installed"
    Write-Host "Unit tests: passed"
    Write-Host "Dataset folder: data\ravdess"
    Write-Host "Training launcher: TRAIN_VOICE_EMOTION.bat"
    Write-Host ""
    Write-Host "Copy the RAVDESS Actor_01 through Actor_24 folders into data\ravdess."
    Write-Host "The WAV data is local and is not stored in GitHub."
    Write-Host "WavLM model weights download on first training run."
    $SetupExitCode = 0
}
catch {
    Write-Host ""
    Write-Host "SETUP FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "See second_pc_setup.log for details."
    $SetupExitCode = 1
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
}

exit $SetupExitCode
