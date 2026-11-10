# SpeechMind

**Automatic multilingual speech understanding for Indian speakers.**

SpeechMind is designed for people who naturally speak English, Hindi, Hinglish and supported Indian languages. It must recognize language mixing automatically; the speaker does not select a language before recording.

The goal is to preserve the original transcript, provide its meaning in English, analyze the words and vocal delivery independently, and combine the evidence into an interpretation of the audio.

## Product principle

**SpeechMind adapts to the speaker, not the speaker to SpeechMind.**

There is no language selector at the beginning of the experience.

## What it does

1. Accepts natural audio (WAV/MP3/M4A).
2. Automatically transcribes English, Hindi and code-mixed Indian speech.
3. Preserves the original transcript.
4. Converts the meaning into English as a separate result.
5. Analyzes text emotion and vocal emotion in parallel.
6. Combines both branches while preserving uncertainty and disagreements.

This is an experimental research system. It does not diagnose anxiety, depression or any medical/psychological condition.

## Current implementation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the stable product specification and component architecture.

| Component | Current implementation |
|---|---|
| ASR | Optional Bodhan Indic-Transcribe API; local Whisper Small fallback |
| Transcript LID | AI4Bharat IndicLID (optional weights); script heuristic fallback |
| Hinglish text emotion | HinEmo-9 character/word TF-IDF + LinearSVC baseline |
| Additional-language text emotion | Fine-tuned MuRIL checkpoint path |
| English text fallback | English-only DistilRoBERTa |
| Voice emotion | Frozen WavLM Base Plus + SVM baseline |
| English meaning | Bodhan Indic-Translate API; limited offline IndicTrans2 fallback |
| Multimodal fusion | Text/voice category mapping with explicit uncalibrated scores |

Model support and real-world accuracy are not established merely because a checkpoint loads. See architecture limitations and evaluation requirements.

## Quick start — Windows

1. Pull the latest `main` branch from GitHub.
2. Run `SETUP_SECOND_PC.bat` if creating the environment for the first time.
3. Install newly added requirements in the activated environment:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Optional: run `SETUP_BODHAN_API_KEY.bat` and add a key from https://console.bodhan.ai/ to the local `.env` file. This enables hosted code-mixed ASR and English translation. Without a key, SpeechMind uses local Whisper and cannot provide the same code-switch-specific ASR/translation path.
5. Optional: run `DOWNLOAD_INDICLID_MODELS.bat` to install native and Romanized Indian transcript language identification.
6. Run SpeechMind:

   ```powershell
   python -m streamlit run app.py
   ```

Cloud mode sends audio to Bodhan for transcription and the transcript for English meaning conversion. The application displays a notice when this mode is configured. Check current fees and provider terms before sending private recordings.

## Text-emotion baseline

Download the HinEmo-9 data:

```powershell
.\DOWNLOAD_HINEMO9_DATA.bat
```

Train the first CPU-friendly Hinglish text baseline:

```powershell
.\TRAIN_HINEMO9_TEXT_EMOTION.bat
```

See [docs/TEXT_EMOTION_TRAINING.md](docs/TEXT_EMOTION_TRAINING.md) for label provenance, limitations and evaluation instructions.

## ASR evaluation

Request access to IITG-HingCoS, construct `data/asr_benchmark/manifest.csv`, then run:

```powershell
.\BENCHMARK_ASR.bat
```

See [docs/ASR_BENCHMARKING.md](docs/ASR_BENCHMARKING.md).

## Data and licensing

See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md). Do not commit local audio, datasets, checkpoints, generated metrics or API keys. Do not redistribute source text from third-party platforms unless the applicable terms permit it.

## Development

- Run unit tests: `python -m unittest discover -s tests -v`
- Compile check: `python -m compileall -q .`
- Voice training: `TRAIN_VOICE_EMOTION.bat` (existing RAVDESS + CREMA-D acted-English baseline only).
- Keep model evaluation separated by task: ASR WER/CER, text-emotion macro-F1, speaker-disjoint voice macro-F1, and joint fusion evaluation.

SpeechMind's product objective and user experience are the project's stable specification. Model candidates can change when evidence supports a better implementation; changing the product objective requires explicit discussion.
