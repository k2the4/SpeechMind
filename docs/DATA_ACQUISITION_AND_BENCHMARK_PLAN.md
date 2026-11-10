# SpeechMind data acquisition and benchmark plan

**Cost rule:** SpeechMind development must use free-to-access datasets and local/open-source tools. Do not purchase a dataset, physical media, API subscription, or hosted inference service without explicit approval. Never download a multi-hundred-gigabyte archive merely because it is available.

## Priority 1 — IndicVoices (free gated access; natural Indian speech)

Official dataset:
https://huggingface.co/datasets/ai4bharat/IndicVoices

License: CC BY 4.0, subject to the dataset card and access terms.

The dataset card describes 23.7K hours of read, extempore, and conversational audio from approximately 51K speakers across 22 Indian languages. It reports 11.2K hours transcribed as of 23 December 2025. The Hugging Face repository is gated, which means a user must sign in and accept/request access. Gated access is not the same as a paid purchase; the current card provides a browser-based access flow. The repository also lists a very large total size (approximately 745 GB), so do not download all files.

### How to access without paying

1. Create or sign in to a free Hugging Face account: https://huggingface.co/join
2. Open https://huggingface.co/datasets/ai4bharat/IndicVoices
3. On the dataset page, click the access/agree button (wording may be "Agree and access repository" or "Request access").
4. Review and accept the dataset conditions. If the request is automatically approved, the page will show that access is granted. If manual approval is required, wait for the dataset owner.
5. Do not download the full-length audio archives (v1–v5). Start with the curated dataset configuration and validation split, and use streaming or a small capped sample for schema inspection.

For Python access, create a read-only Hugging Face access token from https://huggingface.co/settings/tokens. On the SpeechMind PC, authenticate once with:

```powershell
hf auth login
```

Paste the token only into the interactive prompt. Do not commit it, put it in a script, or send it in chat.

The dataset card's example uses:

```python
from datasets import load_dataset
dataset = load_dataset(
    "ai4bharat/IndicVoices",
    "assamese",
    split="valid",
    streaming=True,
)
first_record = next(iter(dataset))
print(dataset)
print(first_record.keys())
```

For SpeechMind, inspect the available configurations and splits before choosing Hindi. Begin with a small validation sample to inspect the audio and JSON metadata fields. The dataset card's total size refers to the overall repository; the project must not pull all 745 GB. Use only a bounded sample for the first benchmark and record exactly which subset was used.

IndicVoices is natural speech and useful for ASR/language coverage. It is not an emotion-labelled dataset and must not be used as voice/text emotion ground truth.

## Priority 2 — Free Hindi-English code-switching evaluation

IITG-HingCoS is a directly relevant Hindi-English code-switching corpus, but the official request page currently lists a paid physical-media charge of INR 50,000 plus GST for Indian organizations. This project will NOT purchase it. It is excluded unless the provider later offers a free academic/research access route in writing.

Official information page:
https://www.iitg.ac.in/eee/emstlab/HingCoS_Database/HingCoS.html

Until a free paired Hinglish corpus is secured, build a small consent-based evaluation set from recordings created specifically for this project, with manually checked transcripts and speaker-disjoint splits. Clearly label it as an internal evaluation set, not a public benchmark.

## Existing voice-emotion data already used

- CREMA-D: 7,442 acted English utterances.
- RAVDESS: 1,440 acted English utterances used in the current training run.

These datasets are already part of the local SpeechMind setup and remain useful for reproducing the WavLM baseline. They do not establish performance on natural Indian speech or Hinglish.

## Text-emotion data

A suitable, openly licensed, human-annotated Hindi-English code-mixed emotion corpus has not yet been verified for this project. Do not assume that a similarly named dataset is the correct one. Before using any third-party corpus, verify the canonical repository, license, annotation provenance, label definitions, and split design.

Social-media text is not equivalent to spontaneous spoken dialogue. Keep semantic emotion and vocal emotion datasets separate unless labels and collection conditions justify a controlled mapping.

## Benchmark protocol

For each ASR candidate:
1. Use the same held-out audio clips and reference transcripts.
2. Normalize text consistently before WER/CER calculation, while preserving a second raw metric.
3. Report overall WER/CER and separate Hindi, English, and code-switch slices where available.
4. Inspect code-switch boundaries, English entity preservation, numbers, negation, and named entities.
5. Keep speakers disjoint across splits where speaker IDs exist.
6. Keep synthetic audio out of the natural-speech test set.

For translation:
1. Use the original ASR output as input.
2. Compare English meaning against human-written references.
3. Include mixed-script and Romanized Hinglish.
4. Check preservation of negation, tense, names, numbers, intent, and idioms.
5. Report translation quality separately from ASR quality.

For emotion:
- Do not infer emotion labels from ASR confidence.
- Keep text and voice annotations separate.
- Use the shared seven-class taxonomy only when source annotations can be mapped reliably.
- Record annotator disagreement and ambiguous examples instead of forcing a label.

## Immediate next action

Use the free Hugging Face access route for IndicVoices. Start with a small Hindi validation subset (streamed, not bulk-downloaded) after inspecting the available configuration names and schema. Then implement a bounded dataset inspection/download script and ASR benchmark runner with Windows launchers. No paid dataset or API is required for this plan.


## Verified Hindi schema and first inspection

The first authenticated three-record inspection succeeded using the streamed Hindi validation split. The returned rows exposed:
- `audio_filepath`: an audio feature that provided FLAC bytes and a source filename.
- `text`: the dataset's transcript text.
- `normalized`: normalized transcript reference.
- `verbatim`: verbatim transcript reference.
- `duration`: audio duration in seconds.
- `lang`: language code.
- `speaker_id`: speaker identifier.
- `scenario` and `task_name`: recording/task context.
- `gender`, `age_group`, `state`, and `district`: metadata fields.

The three inspected examples were short Hindi chunks from the same speaker and an Extempore / District Specific scenario. Adjacent chunk numbering in their audio filenames suggests that a speaker may contribute multiple chunks from one longer recording. The benchmark sampler therefore shuffles the stream and defaults to at most one clip per speaker to improve speaker diversity.

The browser viewer reported approximately 451K rows for the Hindi subset and 5.53K in the validation split at the time the user inspected it. Dataset counts can change with repository revisions; the downloader records the exact subset used in its output summary.

## Bounded local validation workflow

After Hugging Face authentication in the SpeechMind virtual environment:
1. Run `DOWNLOAD_INDICVOICES_HINDI_SAMPLE.bat`.
2. The script streams the Hindi `valid` split, shuffles a 1,024-record buffer, and saves at most 100 FLAC clips with at most one clip per speaker.
3. It stores `manifest.jsonl`, `sample_summary.json`, `ATTRIBUTION.txt`, and audio files under `data/indicvoices/hindi/valid_sample/`.
4. It does not train any model and does not download the full validation split or the full-length archives.

Then run `BENCHMARK_INDICVOICES_HINDI.bat` to evaluate SpeechMind's current local automatic ASR route. Results are stored under `data/indicvoices/hindi/valid_sample/benchmarks/auto/`:
- `predictions.csv`: reference transcripts, ASR hypotheses, per-clip WER/CER, speaker/task metadata, and inference time.
- `summary.json`: corpus-level WER/CER for both normalized and verbatim references, plus scenario breakdowns.

The benchmark can also be run manually with `--model whisper` or `--model indicconformer` to compare the individual local decoders. The automatic route remains the first benchmark because it reflects the current SpeechMind local ASR logic.

Metrics are calculated on a small, shuffled sample from the Hindi validation split. They are a first diagnostic, not a publishable benchmark and not evidence of Hinglish performance. This dataset has speech transcripts but no emotion ground-truth labels in the inspected records, so it is for ASR evaluation only.


## First Hindi baseline results (100-record sample)

The first completed automatic-route benchmark on the 100-record, 100-speaker sample produced:
- Normalized-reference WER: 26.72% (357 word errors / 1,336 reference words).
- Normalized-reference CER: 16.08% (962 character errors / 5,983 reference characters).
- Verbatim-reference WER: 29.85%.
- Verbatim-reference CER: 16.94%.
- Scenario WER: Extempore 21.80%, Conversation 41.90%, Read 16.93%.
- 62 clips were routed to IndicConformer and 38 to Whisper Small.
- On the clips routed to IndicConformer, recomputing corpus-level counts from `predictions.csv` gives WER 16.30% and CER 7.95%.
- On the clips routed to Whisper Small, the corresponding corpus-level WER is 98.24% and CER 75.24%.

This exposes a major weakness in the current whole-clip language router: many Hindi conversational clips, especially short clips, were sent to Whisper and transcribed as English, Urdu, Japanese, Korean, or other non-Devanagari output. The aggregate `fallback_count=0` only means IndicConformer did not fail and fall back; it does not mean every clip used IndicConformer.

The result does **not** establish that IndicConformer will achieve 16.3% WER over the entire Hindi split, because it was only used on the 62 clips chosen by the router. A forced-IndicConformer run over all 100 clips is required for a fair same-sample comparison. Use:
`BENCHMARK_INDICVOICES_HINDI_INDICCONFORMER.bat`.

The evaluator now initializes IndicConformer before per-clip timing and reports model initialization separately. The first original run attributed about 604 seconds of first-use model initialization to a single audio clip, which distorted the latency summary. The next benchmark will report steady per-clip timing separately from initialization.

These are diagnostic results from a bounded 100-record sample, not a publication-grade benchmark. The recordings are Hindi, not Hinglish, and this dataset contains no emotion labels.
