# IndicTrans2 local translation on Windows through WSL

## Why this is separate

The Windows SpeechMind application and the Srota ASR environment remain unchanged.
Only IndicTrans2 runs in a Linux environment under WSL. IndicTransToolkit 1.1.1
states that it is not built or tested for native Windows. The isolated worker
uses the IndicTrans2 Indic-to-English distilled 200M model and communicates with
the Windows app through a persistent JSONL pipe.

## Requirements

- Windows 10/11 with WSL2 and an Ubuntu distribution.
- The SpeechMind project stored on a Windows drive that WSL can access under
  `/mnt/<drive>/...`.
- The existing SpeechMind `.venv` created by `SETUP_SECOND_PC.bat`.

If WSL is not installed, open PowerShell as Administrator and run:

```powershell
wsl --install -d Ubuntu
```

Restart if Windows requests it. Open Ubuntu once and complete its initial Linux
username setup.

## Install and verify

From the SpeechMind PowerShell terminal:

```powershell
.\SETUP_INDIC_TRANS_WSL.bat
```

The launcher installs Python virtual-environment support in Ubuntu and creates
a small bootstrap environment for uv. It uses uv to provision a managed CPython
3.11 runtime for IndicTrans2 instead of Ubuntu's system Python. This avoids
falling back to source builds for the pinned NumPy 2.2 range on Ubuntu releases
whose default Python is newer. The runtime lives at
`~/.venvs/speechmind-indictrans`.

It installs CPU PyTorch, NumPy 2.1-2.2, Transformers 4.57.6,
IndicTransToolkit 1.1.1, and SentencePiece into that WSL-only environment. It
does not modify the Windows `.venv` or `.venv_srota`.

After dependency setup, the launcher runs the real translation smoke test. The
first test downloads `ai4bharat/indictrans2-indic-en-dist-200M` into the WSL
Hugging Face cache. This is a one-time model download; later translations reuse
the cached model and persistent worker.

## App behavior

On Windows, `models.translation.Translator` routes to
`models.wsl_translation_provider.WslIndicTransProvider`. ASR remains in the
existing Windows/Srota path. The original transcript is never overwritten;
the English meaning is returned as a separate field. When WSL or the model is
unavailable, the pipeline retains the original transcript and reports a
translation warning.

Supported source language codes are Hindi, Bengali, Tamil, Telugu, Marathi,
Gujarati, Punjabi, Malayalam, Kannada, and Urdu. Hinglish is sent using the
Hindi FLORES code; code-mixed translation quality must be evaluated separately
before claiming full Hinglish meaning preservation.

## Troubleshooting

- `No usable default WSL distribution`: install Ubuntu with
  `wsl --install -d Ubuntu`, open it once, then rerun setup.
- `python3-venv` or apt errors: open Ubuntu and run
  `sudo apt-get update`, then rerun the setup launcher.
- Python or package installation errors: inspect the first pip/uv error in the
  setup output; the bootstrap and model environments are separate and can be
  rebuilt independently.
- Model download errors: check network access from Ubuntu and rerun
  `SETUP_INDIC_TRANS_WSL.bat`; the Hugging Face cache is retained.
- To validate translation without audio inference, rerun
  `SETUP_INDIC_TRANS_WSL.bat`; it reuses the environment and model cache.
