# Setting up SpeechMind on a second Windows PC

## Goal

Prepare a fresh Windows 10/11 PC with Python, PyTorch, FFmpeg, SpeechMind dependencies, tests, and the local folders needed to run WavLM voice-emotion training.

## What must be installed manually first

Install Git for Windows from https://git-scm.com/downloads/win. During setup:
- Set Visual Studio Code as Git's default editor.
- Choose the option that makes Git available from the command line and third-party software.
- The other defaults are suitable.

Restart VS Code after Git installation.

## Get the project

Open the empty SpeechMind folder in VS Code. In its PowerShell terminal, run:

```powershell
git clone https://github.com/k2the4/SpeechMind.git .
```

The dot means clone into the currently opened, empty folder.

## One-click environment setup

Double-click `SETUP_SECOND_PC.bat` from the project root.

The script:
1. Installs Python 3.11 with WinGet if it is missing.
2. Installs FFmpeg with WinGet if it is missing.
3. Creates the project-local `.venv`.
4. Detects an NVIDIA driver using `nvidia-smi`.
5. Installs a CUDA 12.8 PyTorch/TorchAudio build when an NVIDIA driver is detected; otherwise installs the CPU build.
6. Installs the remaining packages from `requirements.txt`.
7. Verifies core imports, CUDA visibility, and audio dependencies.
8. Runs the unit tests.
9. Creates local dataset, checkpoint, and training-output folders.

The setup log is `second_pc_setup.log`. If Python was just installed but cannot be found by the current terminal, close VS Code, reopen the project, and run the setup file again.

The Torch installation command follows the official PyTorch Windows installation selector. See https://pytorch.org/get-started/locally/.

## Dataset required for training

The RAVDESS WAV files are not in GitHub. Copy the already-downloaded RAVDESS speech dataset (Actor_01 through Actor_24) into:

```
data/ravdess/
```

Expected example:

```
data/ravdess/
  Actor_01/
    03-01-01-01-01-01-01.wav
  Actor_02/
    ...
```

Do not rename the WAV files. The training script recursively scans the folder. The setup script cannot copy files from your other PC; use an external drive or your own OneDrive sync.

## Start training

After the dataset is present, double-click `TRAIN_VOICE_EMOTION.bat`.

The training script downloads `microsoft/wavlm-base-plus` automatically from Hugging Face on its first run, fine-tunes it, selects the best checkpoint using validation macro-F1, and evaluates on the held-out test speakers.

Outputs:
- Checkpoint: `checkpoints/voice_emotion_wavlm/`
- Metrics: `training_outputs/wavlm_ravdess/test_metrics.json`
- Training log: `training_launcher.log`

## Hardware notes

Full WavLM fine-tuning is compute-intensive. CPU mode may take many hours. CUDA detection does not guarantee sufficient VRAM. Keep the PC plugged in and ensure adequate free disk space. The training launcher asks before proceeding on CPU.

## What is not transferred from the old PC

Do not copy the old PC's `.venv` or model cache. They are machine-specific and should be recreated. RAVDESS audio must be copied separately. The Hugging Face WavLM weights download automatically on the new PC.
