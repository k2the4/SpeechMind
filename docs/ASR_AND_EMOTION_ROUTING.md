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

- **Hinglish:** Srota union model is the local default. It is trained for conversational and technical-tutorial Hinglish. Use its native inference prompt by default (`language=None`, empty context); inject context only for actual names or domain terms, not as a long instruction prompt.
- **Hinglish experiment:** Srota-Conv (`moorlee/qwen3-asr-0.6b-hinglish-hiacc-v1`) may be selected in the reference benchmark for conversational recordings. It is not the production default; the union model remains the general-purpose default until a representative held-out evaluation supports changing it.
- **Pure Indic speech:** The existing IndicConformer route remains the multilingual specialist.
- **English:** Whisper remains the English recognition route.

Do not start new ASR fine-tuning based on one recording. First build a held-out evaluation set with verified transcripts, multiple speakers, accents, recording conditions, conversational Hinglish, and technical Hinglish. Compare WER and CER using the same normalisation for each candidate.

Example: compare the local recognizers on the reference clip. The default runs Whisper-small in automatic and forced-Hindi modes, then the general Srota model:

```powershell
.venv\Scripts\python.exe scripts\compare_asr_reference.py --audio "hinglish 2.wav" --whisper-languages auto hi
```

Add `--include-indicconformer` to include the existing IndicConformer Hindi recognizer. The model may need to load its weights on first use. Lower WER/CER is better; a single clip is diagnostic, not a general quality claim. The conversational Srota specialist remains optional because its CPU startup previously timed out on the development machine.

## Automatic Hinglish vocabulary context

When the local router selects Srota, SpeechMind uses Whisper-small on only the final three seconds as a vocabulary scout. It lets Whisper auto-detect the tail language and only extracts exact matches when the tail is detected as English, using a small local allow-list (for example, `layout`, `Figma`, `interface`, or `Python`). The scout transcript is never used as the final transcript. If no allow-listed term is heard, Srota receives an empty context and runs normally. The chosen context, scout transcript, and probe duration are available in diagnostics. This avoids a manual hint field and avoids sending a long instruction prompt to Srota.

The reference benchmark uses the same automatic context by default. Use `--no-auto-context` to measure Srota without the vocabulary scout.

## Srota runtime

The isolated worker follows the model's normal 512-token generation budget. Lowering this ceiling is not an accuracy fix and can truncate the remaining transcript. The inference wrapper also avoids injecting a long custom instruction prompt: the default request uses an empty context, matching the published model-card example.

The isolated worker now selects the runtime automatically:

- CUDA available: use CUDA and BF16 when supported, otherwise FP16.
- CPU-only: use CPU and FP32.
- In automatic mode, a GPU load failure falls back to CPU.
- Override with `SPEECHMIND_SROTA_DEVICE=auto|cpu|cuda`.

FlashAttention 2 is enabled only when the `flash_attn` package is present. This does not make CPU inference faster.

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

## Local CPU runtime tuning and diagnostics

The main SpeechMind process configures PyTorch when the pipeline is
constructed, before model weights are loaded. Its default CPU intra-op thread
limit is 4 and inter-op limit is 2. Override the main-process limit before
launching the app:

```powershell
$env:SPEECHMIND_TORCH_THREADS = "2"
$env:SPEECHMIND_SROTA_TORCH_THREADS = "6"
.venv\Scripts\python.exe -m streamlit run app.py
```

The Srota ASR worker has its own thread pool because it runs in an isolated
Python process. Its default is 4 threads; `SPEECHMIND_SROTA_TORCH_THREADS`
accepts values from 1 to 16. Change it only for a controlled timing comparison.

IndicTrans2 uses the known-compatible decoder generation path with
`use_cache=False`, four-beam search, and a maximum output length of 256 tokens.
The model's custom decoder crashed when KV caching was enabled, so the cache
optimisation has been removed entirely rather than left as a runtime option.
The worker reports `translation_cache_used: false` for transparency.

Validate the installed WSL model after updating the repository with:

```powershell
.venv\Scripts\python.exe scripts\test_indictrans_translation.py
```


The app reports separate ASR, text-emotion, voice-emotion, translation, and
parallel wall-clock timings. Voice diagnostics expose the two highest raw
class scores and their margin. These are uncalibrated model scores; they are
for diagnosing ambiguity, not for converting uncertain predictions into
confident labels. Compare the same recording after a fresh app restart and
after a warm second run before deciding whether a setting improves throughput.

## Runtime policy

All ASR inference in this branch is local: Whisper-small, Srota, and IndicConformer. No API keys, external speech services, or paid inference are required.

## Training policy

Do not fine-tune a new ASR or emotion model during the three-day delivery sprint. Stabilise routing and reliability first. Train only after a versioned evaluation set exists, with speaker-separated train/validation/test splits and per-language reporting. Keep model improvements separate from UI changes so every result is reproducible.
