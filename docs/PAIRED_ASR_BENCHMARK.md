# Paired ASR model comparison

SpeechMind's 40-clip production evaluation showed that the automatic router used different models on different recordings. WER values from those separate subsets are not a controlled model comparison. This benchmark fixes that by running direct candidates against the same selected recordings and references.

## Run the comparison

Prepare HiACC and complete at least one full-pipeline evaluation first. Then run:

```powershell
COMPARE_ASR_MODELS_HIACC.bat
```

The launcher reads the newest `data/benchmarks/full_pipeline/run-*/predictions.csv`, selects 20 clips by default, and uses a deterministic seed and a maximum of one clip per speaker. The exact same selected clips and reference transcripts are used for each candidate.

To run a larger comparison manually:

```powershell
.venv\Scripts\python.exe scripts\benchmark_asr_models_paired.py --limit 40 --seed 42 --max-per-speaker 1
```

To use a particular input file:

```powershell
.venv\Scripts\python.exe scripts\benchmark_asr_models_paired.py --input "data\benchmarks\full_pipeline\run-<timestamp>\predictions.csv" --limit 40
```

## Models tested

| Candidate | Inference mode | Purpose |
|---|---|---|
| Srota auto-context | Whisper tail scout + Srota vocabulary context | Measures the current Hinglish specialist path |
| Srota native | Same Srota checkpoint, no context | Measures whether the automatic vocabulary context improves recognition |
| IndicConformer | Forced Hindi route | Hindi-focused cross-language diagnostic; not assumed to be a Hinglish specialist |
| Whisper-small | Automatic language detection | General-purpose baseline |

The existing production-router transcript from the input predictions file is also included as a historical baseline. It is not re-run as a direct candidate in this benchmark.

## Fairness and timing

- All candidates receive the same decoded 16 kHz mono waveform and are scored against the same alternative-aware reference.
- The sample is selected once before any model inference. Each selection is deterministic for a fixed input CSV, seed and speaker cap.
- Srota is warmed once before scored inference. Worker/model startup and warm-up time are reported separately, not counted as one candidate's first scored clip.
- Srota auto-context records the Whisper vocabulary-scout duration separately and reports effective latency including that scout.
- Model initialization is reported separately from per-recording inference.
- Individual failures are retained as failed predictions; they are not silently removed from the selected sample.
- WER/CER use the same NFKC/lowercase/punctuation normalization and multi-reference scoring helper used by the existing SpeechMind benchmarks.

## Outputs

The benchmark writes:
- `data/hiacc/benchmarks/paired_asr/predictions.csv`: reference, all candidate hypotheses, per-model status, per-clip WER/CER and timings.
- `data/hiacc/benchmarks/paired_asr/summary.json`: micro WER/CER, adult/children and reference-script subgroup metrics, paired wins/ties/losses against the production-router transcript, model initialization and latency.

The Srota auto-context vs native comparison is especially important: it tells us whether the 3-second Whisper scout is worth its latency for the measured accuracy change.

## Decision policy

Do not automatically replace the production router based on a 20- or 40-clip sample. Review:
1. paired micro WER/CER across all clips;
2. adult/children and Devanagari/Latin/mixed-script results;
3. the exact clips where one model wins and another fails;
4. context benefit vs scout latency;
5. failed inference count and median/p90 latency.

Only then make a routing change, and rerun the unchanged reference set afterward. Keep the test split for final reporting; do not use it as a training set. HiACC has documented speaker overlap across its released splits, so this benchmark is not an unseen-speaker generalisation claim.
