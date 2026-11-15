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
| Whisper-small (beam 5) | Automatic language detection | General-purpose baseline |
| Whisper-small fast (beam 1) | Same checkpoint, greedy decoding | Low-latency candidate; accuracy must be checked |

The existing production-router transcript from the input predictions file is also included as a historical baseline. It is not re-run as a direct candidate in this benchmark.

## Fairness and timing

- All candidates receive the same decoded 16 kHz mono waveform and are scored against the same alternative-aware reference.
- The sample is selected once before any model inference. Each selection is deterministic for a fixed input CSV, seed and speaker cap.
- Srota is warmed once before scored inference. Worker/model startup and warm-up time are reported separately, not counted as one candidate's first scored clip.
- Srota auto-context records the Whisper vocabulary-scout duration separately and reports effective latency including that scout.
- Model initialization is reported separately from per-recording inference.
- Individual failures are retained as failed predictions. The main micro WER/CER treats failed calls as empty hypotheses; a separate successful-only metric is provided for debugging. Failures therefore cannot artificially improve the headline score.
- WER/CER use the same NFKC/lowercase/punctuation normalization and multi-reference scoring helper used by the existing SpeechMind benchmarks.

## Outputs

The benchmark writes:
- `data/hiacc/benchmarks/paired_asr/predictions.csv`: reference, all candidate hypotheses, per-model status, per-clip WER/CER and timings.
- `data/hiacc/benchmarks/paired_asr/summary.json`: micro WER/CER, adult/children and reference-script subgroup metrics, paired wins/ties/losses against the production-router transcript, model initialization and latency.

The Srota auto-context vs native comparison is especially important: it tells us whether the 3-second Whisper scout is worth its latency for the measured accuracy change.

## First paired run findings (20 clips)

The first uploaded paired run selected 20 clips (10 adult, 10 children). All model calls completed:
- Srota with automatic context: 16.58% micro WER / 8.08% micro CER.
- Srota without context: identical hypotheses and identical scores on all 20 clips.
- IndicConformer forced Hindi: 38.86% micro WER / 33.56% micro CER overall; it performed much better on the 8 Devanagari-only references (8.45% WER / 4.06% CER) and poorly on mixed-script references (44.33% WER / 39.46% CER).
- Whisper-small automatic: 52.85% micro WER / 33.33% micro CER.
- Historical production-router output: 16.06% micro WER / 8.08% micro CER.

The production router selected IndicConformer on 7 of 8 Devanagari-only clips, Srota on all 9 mixed-script clips, and Whisper on all 3 Latin-only clips. On this small sample this is a promising routing pattern, so it was deliberately left unchanged. The three Latin-only examples are too few for a reliable language-wide conclusion.

The first context scout produced no allow-listed terms on any clip (14 tails were classified non-English, 4 had no allow-listed terms, 2 were too short). Auto-context and native Srota hypotheses were therefore identical on all 20 clips. The scout's median was 5.00 seconds, mean 9.53 seconds, 90th percentile 21.91 seconds and maximum 46.69 seconds. This was unnecessary overhead for this sample. The implementation now runs a fast tail-language check first and only transcribes confidently English tails; the paired benchmark also reuses a single Srota call when context is empty. A subsequent paired run is required to measure the real latency reduction.

These results are an in-domain HiACC diagnostic, not unseen-speaker evidence: HiACC's released train/validation/test splits share speakers according to the dataset/model documentation. Do not generalize the 20-clip values to all Indian speakers or languages.


## Decision policy

Do not automatically replace the production router based on a 20- or 40-clip sample. Review:
1. paired micro WER/CER across all clips;
2. adult/children and Devanagari/Latin/mixed-script results;
3. the exact clips where one model wins and another fails;
4. context benefit vs scout latency;
5. failed inference count and median/p90 latency.

Only then make a routing change, and rerun the unchanged reference set afterward. Keep the test split for final reporting; do not use it as a training set. HiACC has documented speaker overlap across its released splits, so this benchmark is not an unseen-speaker generalisation claim.


## Quick paired benchmark for local recordings

For a small set of recordings that share one manually prepared reference, the
same comparison script can now benchmark multiple audio files in one run. Srota
automatically splits recordings longer than 30 seconds into 28-second chunks
with a two-second overlap, then stitches the chunk transcripts while removing
duplicated overlap words. The original audio is not trimmed.

Example for the Tanisha and Karan recordings:

```powershell
.venv\Scripts\python.exe scripts\compare_asr_reference.py --audio "hinglish female 1.wav" --speaker "Tanisha" --audio "hinglish male 1.wav" --speaker "Karan" --reference "कल हमारी team ने college project के लिए एक नया dashboard design करने पर काम किया। पहले हमने users का feedback देखा, फिर Figma में दो wireframes बनाए। मुझे दूसरा layout बेहतर लगा क्योंकि उसका navigation simple है और सारे important features आसानी से मिल जाते हैं। अब हम prototype को कुछ users के साथ test करके final interface improve करेंगे।" --include-indicconformer --no-auto-context --csv-output "data\benchmarks\tanisha_karan_asr_comparison.csv"
```

The command runs Whisper-small automatic and forced-Hindi decoding, IndicConformer
forced Hindi, and Srota on each complete recording. It writes one row per
recording/model to the CSV, including speaker, duration, reference and hypothesis,
WER, CER, inference time, initialization time, elapsed time, device and status.
The CSV is UTF-8 with BOM for easier opening in Excel. By default the output is
replaced; pass `--append-csv` to add more recordings to an existing report.

To avoid PowerShell or terminal code-page corruption of Hindi references, save
the verified reference transcript in a UTF-8 text file and pass
`--reference-file "path\\to\\reference.txt"` instead of `--reference "..."`.
The script reads UTF-8 (with or without a BOM) directly from the file.

WER/CER values from different recordings are still affected by speaker, pace,
recording conditions and content. Do not interpret a two-speaker sample as a
gender-performance conclusion or as general model-quality evidence.


The paired benchmark now runs both Whisper beam-5 and beam-1 decoding on each selected recording. They share the same loaded Whisper checkpoint and exact audio/reference pair. Compare `whisper_auto` and `whisper_fast` in `summary.json` and `predictions.csv`; do not promote beam-1 to production solely because it is faster.
