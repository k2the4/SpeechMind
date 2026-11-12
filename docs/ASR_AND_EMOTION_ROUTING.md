# SpeechMind routing and reliability decisions

This document records the current implementation decisions for the three-day prototype. It is intentionally explicit about what is implemented, what is only a candidate, and what still requires evaluation.

## Product flow

SpeechMind accepts a natural speech recording, identifies the language, transcribes the original speech, analyses the words and the original audio independently for emotion, combines usable emotion evidence, and presents the meaning and emotional context in English.

The original transcript is retained. Translation must not overwrite or silently rewrite what the speaker said.

## Language scope

The initial target set is:

- Hindi
- English
- Hindi-English code-switching (Hinglish)
- Bengali
- Tamil
- Telugu
- Marathi

The current ASR routing and IndicTrans2 FLORES mapping contain routes for these languages. ASR or translation support does not automatically mean that the installed text-emotion checkpoint supports the same language. Text emotion is run only when the checkpoint metadata declares that language (or Hinglish) as supported; otherwise that branch is reported unavailable. Do not claim complete end-to-end emotion support for a language until its text and voice checkpoints have been evaluated for it.

## ASR provider policy

- **Hinglish:** Srota union model is the local default. It is trained for conversational and technical-tutorial Hinglish.
- **Hinglish experiment:** Srota-Conv (`moorlee/qwen3-asr-0.6b-hinglish-hiacc-v1`) may be selected in the reference benchmark for conversational recordings. It is not the production default; the union model remains the general-purpose default until a representative held-out evaluation supports changing it.
- **Pure Indic speech:** The existing IndicConformer route remains the multilingual specialist.
- **English:** Whisper remains the English recognition route.

Do not start new ASR fine-tuning based on one recording. First build a held-out evaluation set with verified transcripts, multiple speakers, accents, recording conditions, conversational Hinglish, and technical Hinglish. Compare WER and CER using the same normalisation for each candidate.

Example: run the conversational specialist on the reference clip without changing the production model:

```powershell
.venv\Scripts\python.exe scripts\compare_asr_reference.py --srota-model moorlee/qwen3-asr-0.6b-hinglish-hiacc-v1 --skip-whisper
```

The script prompts for an audio path when `--audio` is omitted. Lower WER/CER is better; a single clip is diagnostic, not a general quality claim.

## Srota runtime

The isolated worker now selects the runtime automatically:

- CUDA available: use CUDA and BF16 when supported, otherwise FP16.
- CPU-only: use CPU and FP32.
- In automatic mode, a GPU load failure falls back to CPU.
- Override with `SPEECHMIND_SROTA_DEVICE=auto|cpu|cuda`.

FlashAttention 2 is enabled only when the `flash_attn` package is present. This does not make CPU inference faster; hosted inference remains a separate deployment option to evaluate.

## Meaning and translation

IndicTrans2 is an Indic-to-English translation model. For mixed Hindi-English input, the existing route is best-effort: it sends the mixed transcript through the Hindi-to-English path. The translated meaning must be displayed separately from the original transcript and accompanied by the mixed-script caveat.

If the selected ASR does not expose a usable transcript-quality score, the application now says so. Unknown ASR confidence is not evidence that the transcript is definitely poor, but it is also not evidence that the transcript is correct. Users must be able to review the original transcript before relying on its meaning or text-emotion result.

Romanized Hindi is not passed directly to IndicTrans2. A transliteration stage is required before that route can be claimed as supported.

## Emotion branches and fusion

- **Text branch:** MuRIL analyses the original transcript, not the English translation. The route is gated by checkpoint-declared language support.
- **Voice branch:** WavLM/wav2vec2-based analysis receives the preprocessed original audio independently of ASR.
- **Fusion:** Known low ASR quality reduces the text branch's influence. When ASR quality is unavailable, the branches receive equal initial weight rather than treating missing as low.
- **Ambiguous voice output:** If the top two voice scores are too close, the voice label is marked uncertain and excluded from fusion. If text evidence is usable, the overall interpretation is explicitly text-only. If no reliable branch remains, the system abstains.
- **Stable disagreement:** When both branches produce stable but different labels, the system does not manufacture a combined emotion.

All model scores and fusion weights remain uncalibrated until validated on paired, held-out Indian-language speech with human emotion annotations. Emotion output is descriptive and is not a diagnosis.

## Hosted ASR candidate

Sarvam Saaras v4 is a candidate for a later hosted deployment comparison because its API documents transcription, translation, verbatim, transliteration, and code-mixed modes for Indian languages. It is not integrated into this branch. A hosted provider would transmit recordings outside the local application and may incur usage charges, so it must remain an explicit deployment choice with disclosure and credentials rather than a silent fallback.

Documentation: https://docs.sarvam.ai/api/api-guides-tutorials/models/saaras

## Training policy

Do not fine-tune a new ASR or emotion model during the three-day delivery sprint. Stabilise routing and reliability first. Train only after a versioned evaluation set exists, with speaker-separated train/validation/test splits and per-language reporting. Keep model improvements separate from UI changes so every result is reproducible.
