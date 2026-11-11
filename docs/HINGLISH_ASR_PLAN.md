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


## Same-clip IndicConformer diagnostic

A diagnostic comparator is now available:
- Launcher: `BENCHMARK_HIACC_INDICCONFORMER.bat`
- Script: `scripts/compare_hiacc_indicconformer.py`
- Input: the existing `data/hiacc/benchmarks/srota/predictions.csv`
- Output: `data/hiacc/benchmarks/indicconformer/summary.json` and `predictions.csv`

It reuses the exact 20 audio paths and reference transcripts from the completed Srota run, forces Hindi decoding through the current IndicConformer model, and uses the same SpeechMind WER/CER normalization utility. It deliberately does not use Whisper fallback: if IndicConformer cannot initialize or infer, the diagnostic stops rather than silently measuring a different model. This is a cross-language stress diagnostic, not a fair contest between two Hinglish specialists, because IndicConformer is the current Hindi baseline and is not designed as the Hinglish specialist.

Run this launcher only after pulling the latest feature branch. It uses the existing `.venv`; it does not install packages or rerun dataset preparation/Srota setup. Review its output before deciding whether a larger controlled Hinglish evaluation is worthwhile.


## First same-clip diagnostic comparison (2026-11-11)

The initial Srota smoke-test outputs and the forced-Hindi IndicConformer diagnostic were compared by `audio_path`; all 20 rows matched exactly.

| Measure | Srota | IndicConformer (forced Hindi) |
|---|---:|---:|
| Samples | 20 | 20 |
| Micro WER | 16.96% | 36.61% |
| Micro CER | 10.04% | 32.82% |
| Mean clip inference | 6.10 s | 0.43 s |
| Median clip inference | 5.763 s | 0.373 s |

At clip level, Srota had lower WER on 12 clips, IndicConformer on 3, with 5 ties. Srota had lower CER on 13, IndicConformer on 3, with 4 ties.

Interpretation:
- Srota is the stronger current Hinglish candidate on this small diagnostic sample.
- IndicConformer is much faster on CPU, but forced-Hindi decoding often renders English speech as Devanagari phonetics. This is expected to be a poor fit for mixed-language speech and is not a valid reason to replace the Hinglish specialist.
- This comparison does not establish production accuracy. It is only 20 utterances from one corpus, and the two models have different intended language scopes.
- The next useful measurement is SpeechMind's actual automatic ASR route on these same 20 clips. That path uses Whisper language detection, IndicConformer for confidently detected Indic languages, and Whisper for other routes. Measure it without changing routing or the application environment.


## Measure the current automatic SpeechMind ASR route

The next diagnostic launcher is `BENCHMARK_HIACC_SPEECHMIND_AUTO.bat`, backed by `scripts/benchmark_hiacc_speechmind_auto.py`.

It reads the same saved 20-clip Srota prediction CSV and runs the current `SpeechRecognizer.transcribe()` automatic path. That means Whisper performs language detection and the existing route chooses IndicConformer for confidently detected Indic languages or Whisper for other cases. The script records detected language, model route, fallback, transcript, WER/CER and inference time per clip.

Run after pulling the latest feature branch. It uses the existing `.venv`, installs nothing, and does not change production code or routing. The first clip routed to IndicConformer may include lazy model initialization in its timing. Results are saved to `data/hiacc/benchmarks/speechmind_auto/`.

This is the most useful next check: it measures the system users currently have, rather than only comparing isolated candidate models. Once complete, compare its per-clip output and routing against Srota. If the automatic route underperforms on Hinglish, the next engineering task is a language-span/code-switch-aware routing strategy—not simply replacing every utterance with Srota.


## Current automatic SpeechMind route comparison (2026-11-11)

The current automatic ASR route was run on the exact same 20 HiACC clips used by the Srota smoke test. It uses Whisper primary-language detection, then routes confidently detected Indic speech to IndicConformer and other speech to Whisper Small.

| Measure | SpeechMind automatic | Srota |
|---|---:|---:|
| Samples | 20 | 20 |
| Micro WER | 27.68% | 16.96% |
| Micro CER | 22.88% | 10.04% |
| Mean clip inference | 3.40 s | 6.10 s |

SpeechMind selected IndicConformer for 14 clips and Whisper Small for 6. Whisper detected Hindi on 15 clips, English on 4, and Bengali on 1. At clip level, Srota had lower WER on 11 clips, SpeechMind automatic on 4, and 5 tied; Srota had lower CER on 13, SpeechMind automatic on 4, and 3 tied.

Interpretation and decision:
- The existing primary-language routing is not adequate for Hinglish. A Hindi language guess routes a mixed Hindi-English utterance to IndicConformer, which does not preserve code-switching as well as the Hinglish-specialized candidate on this sample.
- Srota is now the leading Hinglish ASR candidate based on the available same-clip evidence. This is still a small diagnostic, not production validation.
- Do not replace the pure-Hindi IndicConformer baseline solely from this Hinglish test. Keep language-specialist routing as a goal, but distinguish mixed Hindi-English from pure Hindi automatically.
- Srota's inference environment must remain isolated: its qwen-asr dependency requires Transformers 4.57.6, while the main application pins Transformers 4.46.3. Do not install Srota dependencies into the main .venv.
- Next implementation phase: add an isolated Srota inference provider behind the existing ASR interface, preserving the main environment. Add automatic mixed-speech routing and explicit model/fallback metadata. Evaluate pure Hindi and English separately before finalizing specialist selection. Do not repeat the completed 20-clip smoke test unless the model, routing hypothesis, or evaluation set changes.

Caveats: the 20-clip HiACC test sample is diagnostic only; it does not establish production accuracy or unseen-speaker generalization. The results are not evidence for Bengali or other regional-language support.


## First Srota integration implementation (2026-11-11)

The main ASR interface now has an isolated Srota probe for automatic code-switch handling:
- `models/srota_provider.py` manages one persistent worker process and sends audio to it through a JSON-lines protocol.
- `scripts/srota_asr_worker.py` loads Qwen3-ASR inside `.venv_srota`; the main `.venv` and its Transformers 4.46.3 pin remain untouched.
- Hindi-dominant audio is probed with Srota. Its output is selected only when both Devanagari and Latin scripts are present; otherwise the existing IndicConformer Hindi route remains the pure-Hindi baseline.
- An uncertain English primary-language detection is also probed, because code-switched speech can be classified as English. Clear English-only audio remains on Whisper.
- If the isolated environment is missing, the clip exceeds Srota's current 30-second limit, or the worker fails, SpeechMind retains its existing ASR route and records a diagnostic message in the application log.
- The worker remains loaded between requests, avoiding repeated model initialization for each clip.

This is an initial automatic routing heuristic, not a validated language-span detector. Srota script output is used as a code-switch signal; Romanized Hindi that Srota emits entirely in Latin script may not be detected by this first version. The English language-score threshold is provisional and is not a calibrated probability. The next user-side action is a one-clip integration smoke test before running the full application.
