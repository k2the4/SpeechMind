# Setting up SpeechMind on a second Windows PC

## Supported setup

- Windows 10/11
- Python 3.11 (recommended; Python 3.10 is accepted)
- A virtual environment local to this repository
- PyTorch selected from the packages in `requirements.txt`
- CUDA is used automatically only when the installed PyTorch build and NVIDIA driver support it; otherwise training runs on CPU

## One-time setup

1. Install Git for Windows if you want to clone/pull the repository.
2. Install Python 3.11 and enable the Python Launcher (`py.exe`).
3. Clone or download this repository.
4. Double-click `SETUP_SECOND_PC.bat` from the project root.
5. Wait for the setup checks to finish. The full log is `second_pc_setup.log`.

The setup script creates `.venv`, installs `requirements.txt`, checks core imports and CUDA visibility, runs the unit tests, creates the local dataset/output folders, and checks for FFmpeg. It does not install anything globally into system Python.

## Dataset required for training

The RAVDESS audio files are not part of GitHub. Copy the existing RAVDESS speech dataset (the actor folders and their WAV files) into:

```
data/ravdess/
```

The resulting layout should look like:

```
data/ravdess/
  Actor_01/
    03-01-01-01-01-01-01.wav
    ...
  Actor_02/
    ...
```

The training script recognizes official RAVDESS speech filenames recursively. Do not rename the WAV files. The setup script does not copy data from another PC or download the dataset.

## Start training

After the dataset is copied, double-click `TRAIN_VOICE_EMOTION.bat`. It uses the local `.venv`, checks PyTorch/CUDA, and asks before CPU training. The WavLM checkpoint is saved separately at `checkpoints/voice_emotion_wavlm`; test metrics are saved at `training_outputs/wavlm_ravdess/test_metrics.json`.

## Hardware note

WavLM full fine-tuning is resource-intensive. Setup reports whether PyTorch sees CUDA, but that alone does not guarantee enough VRAM or RAM. If this PC has no compatible NVIDIA GPU, training may be very slow. Keep the PC plugged into power and ensure adequate free disk space before starting.

## Troubleshooting

- `second_pc_setup.log`: dependency, import, test, and FFmpeg diagnostics.
- `training_launcher.log`: training-time diagnostics.
- Do not copy `.venv`, model checkpoints, or cache folders from the old PC. Recreate the environment on this PC using the setup script.
