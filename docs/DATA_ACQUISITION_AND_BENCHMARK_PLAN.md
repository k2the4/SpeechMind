# SpeechMind data acquisition and benchmark plan

This document records the data sources to evaluate for the fixed SpeechMind objective. Do not download every dataset in bulk. Start with the smallest directly relevant evaluation set, verify its access terms, and preserve a held-out test split.

## Priority 1 — IITG-HingCoS (Hindi-English code-switching ASR)

Official project page:
https://www.iitg.ac.in/eee/emstlab/HingCoS_Database/HingCoS.html

The IIT Guwahati EMST Lab page describes 25,988 Hinglish text sentences and 25 hours of matching speech for 9,251 code-switched sentences, recorded in realistic environments using landline and mobile phones. The page provides a database request link rather than an unrestricted direct download.

Why this is first:
- It directly targets Hindi-English code-switching.
- It contains paired speech and transcripts.
- It is a much more manageable starting point than multi-hundred-gigabyte corpora.

Action:
1. Open the official page.
2. Use its database request link and follow the lab's access process.
3. Keep the original archive and accompanying transcript/metadata unchanged.
4. After access is granted, inspect the archive layout before running any preparation script. Do not assume a folder or annotation format before seeing the files.
5. Keep speaker identities available so train/validation/test splits can be speaker-disjoint.

Do not publish the corpus or redistribute its audio. Follow the provider's access and usage terms.

## Priority 2 — IndicVoices (natural Indian speech)

Official dataset card:
https://huggingface.co/datasets/ai4bharat/IndicVoices

The dataset card describes 23.7K hours from approximately 51K speakers across 22 Indian languages, including read, extempore, and conversational speech. It reports 11.2K hours transcribed as of 23 December 2025 and a CC BY 4.0 license. Hugging Face access is gated/approval-based. The full-length audio archives total approximately 745 GB.

Why it matters:
- It adds natural Indian speech and speaker/accent diversity.
- It can support ASR benchmarking and language coverage work.
- It is not an emotion-labelled corpus and must not be treated as one.

Action:
1. Request access through the Hugging Face dataset page.
2. Start with one language and a small validation slice, not all 745 GB.
3. Prefer the curated Hugging Face metadata/transcribed split initially.
4. Download full-length archives only if a specific experiment requires them.
5. Respect CC BY 4.0 attribution and any access requirements.

## Existing voice-emotion data already used

- CREMA-D: 7,442 acted English utterances.
- RAVDESS: 1,440 acted English utterances used in the current training run.

These datasets remain useful for reproducing the current WavLM baseline and checking the training pipeline. They do not establish performance on natural Indian speech or Hinglish.

## Text-emotion data

HinEmo-9 is a candidate Hindi-English code-mixed text emotion dataset:
https://huggingface.co/datasets/

Before using any third-party emotion dataset, verify the exact dataset repository, annotation scheme, license, split design, and whether labels are human-annotated or model-generated. Social-media text is not equivalent to spontaneous spoken dialogue. Use human-annotated examples for primary evaluation.

## Benchmark protocol

For each ASR candidate:
1. Use the same held-out audio clips and reference transcripts.
2. Normalize text consistently before WER/CER calculation, while preserving a second raw metric.
3. Report overall WER/CER and separate Hindi, English, and code-switch slices.
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
- Use the same seven-class taxonomy only when the source annotation can be mapped reliably.
- Record annotator disagreement and ambiguous examples instead of forcing a label.

## Current next action

Obtain IITG-HingCoS access first. Once the archive is available locally, provide the archive/folder or its file listing. Then add a preparation script against the actual schema, a reproducible ASR benchmark runner, and Windows batch launchers. No dataset downloader is included yet because the first corpus requires access approval and its received archive structure must be inspected rather than guessed.
