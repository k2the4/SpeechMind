# SpeechMind data acquisition and benchmark plan

**Cost rule:** Use free-to-access datasets and local/open-source tools. Do not purchase datasets, physical media, API subscriptions, or hosted inference services without explicit approval. Do not download multi-hundred-gigabyte archives for exploratory work.

## Hindi ASR baseline — complete for now

The IndicVoices Hindi validation experiment is complete. On the same 100 recordings:
- Current local automatic routing: 26.72% normalized WER / 16.08% normalized CER.
- Forced IndicConformer: 14.22% normalized WER / 5.11% normalized CER.
- Whisper Small with automatic language detection: 73.13% normalized WER / 47.50% normalized CER.

Decision: keep IndicConformer as SpeechMind's current Hindi baseline. Do not spend more time on repeated Hindi Whisper benchmarks unless a materially different candidate or a new failure hypothesis appears. This does not claim Hinglish accuracy.

## Hinglish ASR — active development track

### Dataset: HiACC

Official Zenodo record:
https://zenodo.org/records/15551669

The archive is 531.6 MB and contains 5.24 hours of segmented Hinglish speech: 3,318 adult utterances and 1,858 child utterances. The corpus contains manually checked transcripts and token-level Hindi/English code-switch annotations. Its directory layout is described as Adult/Children, each containing audio, transcripts, metadata, and annotations with train/test/validation splits.

The Zenodo record currently lists Creative Commons Attribution 4.0. The dataset article's access description also contains a CC BY-NC statement, so there is a license-description discrepancy. For this academic prototype, keep use limited to internal, non-commercial research and retain attribution/citation. Do not redistribute the archive or derived data until the authors clarify the discrepancy.

The dataset is not a text- or voice-emotion corpus. Use it only for ASR and code-switch evaluation.

### Leading candidate: Srota

Model:
https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish

Srota is an Apache-2.0 fine-tune of Qwen3-ASR-0.6B intended specifically for conversational and tutorial Hinglish. It preserves mixed Devanagari + Latin script (for example, Hindi words in Devanagari and English words in Latin script). Its model card reports 15.85% WER on the HiACC conversational test set and 35.06% on the OpenSLR-104 tutorial set, using its own stated normalization. These are the model author's results, not SpeechMind results.

Important limitations disclosed by the model author:
- The model is intended for Hindi-English code-switching, not pure Hindi or pure English.
- Its full speech model is approximately 0.8B parameters.
- The documented quickstart recommends BF16 and a single GPU.
- The author notes HiACC train/validation/test share speakers, so the reported HiACC test result is in-domain rather than a strict unseen-speaker result.
- Accent and domain coverage are limited.

For SpeechMind's CPU-only GT 730 machine, the project uses a separate optional environment and requests float32 CPU inference. This is an experimental compatibility path; speed and memory use must be measured locally. Do not change the main SpeechMind .venv or its Transformers 4.46.3 installation.

## Next workflow

1. Run `PREPARE_HIACC_DATASET.bat` to download the 531.6 MB archive from Zenodo, verify its MD5, extract it under `data/hiacc/raw`, and build a test-split manifest.
2. Run `SETUP_HINGLISH_ASR_ENV.bat` to create a separate `.venv_srota` and install the Srota inference package plus CPU PyTorch.
3. Run `BENCHMARK_HIACC_SROTA.bat` for an initial 20-utterance smoke benchmark, balanced across adults and children where available.
4. Review the output before increasing the test sample. Results are written to `data/hiacc/benchmarks/srota/summary.json` and `predictions.csv`.

The setup script and benchmark are isolated from the main SpeechMind environment. No paid API or paid dataset is used.

## IndicVoices — regional-language ASR

Official dataset:
https://huggingface.co/datasets/ai4bharat/IndicVoices

IndicVoices remains useful for natural speech and speaker/accent diversity across Indian languages. The local Hindi validation schema was inspected, and a 100-clip sample was downloaded. It is an ASR corpus, not an emotion-labelled corpus.

Do not download its full 745 GB archive. Any expansion to Bengali, Marathi, Tamil, Telugu, or Punjabi should use a small language-specific streamed sample and a separate WER/CER benchmark.

## Existing voice-emotion data

- CREMA-D: 7,442 acted English utterances.
- RAVDESS: 1,440 acted English utterances used in the current training run.

These datasets remain useful for reproducing the WavLM baseline. They do not establish performance on spontaneous Indian speech or Hinglish.

## Text-emotion data

A suitable, openly licensed, human-annotated Hinglish speech-transcript emotion corpus has not yet been verified. HiACC provides ASR transcripts and code-switch tags, not emotion labels. Do not derive emotion labels from transcription or speaker demographics.

## Evaluation rules

- ASR: report corpus-level WER/CER, raw and normalized variants, language/script behaviour, adult/child subgroup, and inference time.
- Keep the dataset's official test split intact for evaluation.
- Never mix test utterances into training or validation.
- Record the model revision, package versions, normalization, sample selection seed, and exact clip IDs.
- Clearly distinguish model-card results from SpeechMind's own results.
- Do not compare WER values across papers unless normalization and test splits match.
- For any future model training, keep speakers disjoint across train/validation/test whenever source metadata permits.

## HiACC references with multiple accepted transcriptions

Some HiACC transcript rows contain alternative accepted spellings in one field,
separated by three or more spaces. Dataset preparation preserves the original
field. ASR evaluation splits these alternatives only for scoring and selects the
best matching reference independently for WER and CER. The per-row output
records the selected alternative index. Do not concatenate alternatives into
one reference sentence: doing so incorrectly counts an exact match to one
accepted spelling as insertions/deletions against another.
