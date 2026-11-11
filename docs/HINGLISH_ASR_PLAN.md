# Hinglish ASR: next implementation phase

## Product requirement

SpeechMind users speak naturally. They do not select Hindi, English, or Hinglish before recording. The ASR layer must preserve Hindi-English code-switching within a single utterance and output natural mixed-script text where appropriate.

For example:
"मेरा favourite festival Diwali है"

The transcript is the original speech. English meaning is a separate downstream output; it must not overwrite or replace the original transcript.

## Current language-specialist decisions

- Hindi: IndicConformer is the current local baseline. On the same 100-clip IndicVoices Hindi validation sample it measured 14.22% normalized WER and 5.11% normalized CER. This is a bounded Hindi result, not Hinglish evidence.
- English: Whisper remains a candidate; the current SpeechMind English route still needs its own representative validation.
- Hinglish: Srota is the leading local candidate to evaluate. It is a fine-tuned Qwen3-ASR-0.6B model intended for Hindi-English code-switching and native mixed Devanagari + Latin output.

No more Hindi Whisper benchmark runs are planned unless a new model or specific hypothesis justifies the time.

## Candidate: Srota

Model: https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish

- Model license: Apache-2.0, according to the model card.
- Approximate size: 0.8B total speech-model parameters.
- Intended use: conversational and tutorial Hinglish.
- Output style: Hindi in Devanagari, English in Latin script.
- Model-card WER: 15.85% on the HiACC conversational test set and 35.06% on OpenSLR-104 tutorial speech. These are author-reported values and are not SpeechMind measurements.
- The model card recommends BF16 on a GPU. The SpeechMind PC has a GT 730 and 16 GB RAM, so the initial local diagnostic uses CPU float32. CPU speed and memory use are not yet established.
- The model card warns that HiACC train/validation/test share speakers, making its reported HiACC test result an in-domain score rather than a strict unseen-speaker result.

This candidate is unusually well matched to SpeechMind's core Hinglish requirement. It is a candidate to validate, not a final production decision.

## Dataset: HiACC

Zenodo: https://zenodo.org/records/15551669
DOI: 10.5281/zenodo.15551669

The record contains a 531.6 MB archive with 5.24 hours of segmented Hinglish speech: 3,318 adult utterances and 1,858 child utterances. It includes manually checked transcripts and code-switch annotations. The published corpus layout describes Adult and Children directories with audio, transcripts, metadata and annotations, each split into train/test/validation.

The Zenodo record lists CC BY 4.0. The accompanying dataset article also includes a CC BY-NC statement in its access description, so the license descriptions conflict. Until the authors clarify the discrepancy, keep use limited to internal non-commercial academic research, preserve attribution, and do not redistribute the corpus or derived data. The model weights themselves are separately identified as Apache-2.0 by the Srota model card.

HiACC is for ASR/code-switch evaluation only. It is not an emotion-labelled dataset.

## Separate environment

The main SpeechMind environment pins Transformers 4.46.3 for the existing IndicConformer/voice stack. The Srota qwen-asr 0.0.6 inference package pins Transformers 4.57.6. These must not be installed into the same environment.

Use:
- Main application environment: .venv
- Srota candidate environment: .venv_srota

The Srota setup BAT installs CPU-only PyTorch and qwen-asr in .venv_srota; it must not modify .venv.

## First local experiment

1. PREPARE_HIACC_DATASET.bat
   - Downloads the public 531.6 MB Zenodo archive.
   - Verifies its MD5 checksum.
   - Extracts it under data/hiacc/raw.
   - Builds a paired audio/transcript manifest and reports test-split counts.

2. SETUP_HINGLISH_ASR_ENV.bat
   - Creates isolated .venv_srota.
   - Installs CPU PyTorch and qwen-asr==0.0.6.
   - Verifies the package import.
   - Does not alter the existing SpeechMind .venv.

3. BENCHMARK_HIACC_SROTA.bat
   - Loads the Srota model on CPU in float32.
   - Runs an initial 20-utterance smoke test, balanced across adult/children where possible and capped at one clip per speaker where speaker IDs are available.
   - Saves summary.json and predictions.csv in data/hiacc/benchmarks/srota.

If the first 20-clip smoke test is too slow or runs out of RAM, stop it and retain the logs. Do not change the main SpeechMind environment or purchase a hosted inference service.

## After the smoke test

Review:
- Whether Srota loads successfully on the target PC.
- Model initialization time and per-clip CPU inference time.
- WER/CER against the HiACC human transcripts.
- Mixed-script preservation.
- Adult versus children breakdown.
- Any failures, empty outputs or out-of-memory errors.

Only after this review should Srota be integrated into the production ASR layer. The production interface remains automatic; users never choose a language. The eventual routing policy must distinguish pure Hindi, pure English and mixed Hindi-English speech without requiring a user setting, and must be tested on all three groups.


## Initial Srota smoke-benchmark result (2026-11-11)

The first local run completed successfully on the target Windows PC. This is a diagnostic smoke test, not a full corpus evaluation.

| Measure | Result |
|---|---:|
| Dataset / split | HiACC test |
| Samples | 20 (10 adult, 10 children) |
| Sampling | At most one clip per speaker |
| Model | `moorlee/qwen3-asr-0.6b-hinglish` |
| Runtime | CPU, float32 |
| Overall WER | 16.96% |
| Overall CER | 10.04% |
| Adult WER / CER | 20.87% / 13.32% |
| Children WER / CER | 12.84% / 6.76% |
| Mean clip inference | 6.10 s |
| Median clip inference | 5.763 s |
| Model initialization | 6.588 s (weights already cached) |
| Total elapsed | 138.368 s |

There were 8 Devanagari-only, 2 Latin-only, and 10 mixed Devanagari+Latin hypotheses. This is an initial indication that the model emits mixed-script output; it is not yet a measured script-preservation accuracy score.

Interpretation and caveats:
- The 20-clip sample is too small to establish production accuracy or compare adult and child performance reliably.
- HiACC speakers overlap across train/validation/test in the published corpus; this smoke test does not establish unseen-speaker generalization.
- One reference contains a duplicated utterance, which inflates its error score. Romanized names versus Devanagari reference spellings and tokenization differences also affect WER/CER.
- HiACC evaluates Hinglish ASR/code-switching only. It does not evaluate emotion recognition, translation quality, or other Indian languages.
- The benchmark uses SpeechMind's NFKC/lowercase/punctuation normalization. Preserve the raw predictions and summary alongside any future comparable runs.

Next decision: run the existing SpeechMind Hindi ASR baseline on the exact same 20 audio files only if a compatible, isolated benchmark path can be established without altering the main `.venv`. Compare per-clip outputs and identical normalization. Do not integrate Srota into production routing on the basis of this smoke test alone.
