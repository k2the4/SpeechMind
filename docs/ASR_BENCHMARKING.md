# SpeechMind ASR Benchmark

## Why this benchmark exists

SpeechMind must choose its speech recognizer using human-transcribed code-mixed Indian speech, not only a vendor's published results. The benchmark compares:
- Local OpenAI Whisper Small.
- Bodhan Indic-Transcribe, if a local `BODHAN_API_KEY` is configured.

The benchmark reports aggregate WER and CER and saves every reference/prediction pair for error analysis.

## 1. Create the manifest

The file is `data/asr_benchmark/manifest.csv`. Start from `docs/asr_manifest_template.csv`.

Required columns:
- `audio_path`: WAV/MP3/M4A path, absolute or relative to the repository.
- `reference`: human-verified transcription for that exact audio file.

Optional columns: `language`, `split`, `speaker_id`.

Use real, consented or appropriately licensed recordings and human-verified references. Include:
- Natural Hinglish with Romanized Hindi and English.
- Hindi in Devanagari mixed with English.
- English with Indian accents.
- Hindi-only.
- Examples from Bengali, Marathi, Tamil, Telugu and Punjabi as they become available.

Keep speaker IDs and split assignments. A speaker must not appear in both model-training and held-out evaluation data where supervised training is involved. For a fixed ASR benchmark, preserve an untouched test set.

Do not create reference transcripts by copying one ASR model's output. That would measure agreement with a model rather than recognition accuracy.

## 2. Obtain code-switched benchmark audio

Request IITG-HingCoS access from:
https://www.iitg.ac.in/eee/emstlab/HingCoS_Database/HingCoS.html

The dataset has 25 hours of matching audio/text and 9,251 utterances. Access is request-based. Follow the dataset owner's conditions and original split rules.

## 3. Run the comparison

From the SpeechMind root folder:

```powershell
.\BENCHMARK_ASR.bat
```

The benchmark uses both recognizers when Bodhan is configured. Without an API key, only local Whisper is evaluated. Bodhan is not silently substituted for Whisper in the benchmark; an explicit Bodhan benchmark failure is reported as a skip/error. Results are grouped overall, by the manifest's language column and by its split column. Mean ASR latency is reported as well.

## 4. Outputs

- `training_outputs/asr_benchmark/metrics.json`: overall and per-language/per-split WER/CER plus latency by backend.
- `training_outputs/asr_benchmark/predictions.csv`: source reference, model transcript, clip metadata and per-clip errors.

WER and CER are calculated after Unicode normalization, lowercasing and punctuation removal. For mixed scripts, preserve the intended orthography in references. If evaluating transliteration equivalence, define a separate normalization protocol and report it separately; do not hide script errors by aggressive normalization.

## 5. Model selection rule

Select the recognizer based on:
1. Hinglish/code-switch WER and CER.
2. Per-language results, not just aggregate results.
3. Word substitutions at language-switch boundaries.
4. Proper nouns, numbers and common Indian-English terms.
5. Latency, API cost and privacy requirements.

Do not declare the best model until the same untouched, human-transcribed test set has been run through all candidates.
