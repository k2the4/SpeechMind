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
