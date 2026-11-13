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

## Hinglish vocabulary context policy

Automatic Whisper tail-vocabulary scouting is **disabled by default in the production app**. Hindi-dominant and uncertain-language Srota requests use the model's native empty-context path; they do not run an additional Whisper pass just to collect technical terms.

When the router has already decoded a Whisper transcript to check a high-confidence English result for Hinglish, it may extract exact allow-listed terms from that existing transcript before a Srota retry. This reuses work that has already happened and does not launch another Whisper inference.

The extra three-second Whisper tail scout remains available for controlled experiments only. To opt in for a run, set this before launching Streamlit:

```powershell
$env:SPEECHMIND_ENABLE_ASR_CONTEXT_SCOUT = "1"
.venv\Scripts\python.exe -m streamlit run app.py
```

Leave the variable unset (or set it to `0`) for the production default. The scout first checks the tail language and transcribes it only when it is confidently English. Its output is used only as a small allow-listed vocabulary hint, never as the final transcript. The paired benchmark can still compare automatic-context and native-context Srota variants explicitly.

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

## Runtime diagnostics

The isolated Srota worker reports model/runtime initialization separately from
the timed `model.transcribe` call. Diagnostics expose
`asr_candidate_initialization_seconds`, `asr_candidate_inference_seconds`,
and `asr_candidate_worker_request_seconds`. The inference field now excludes
first-load model initialization.

The IndicTrans2 WSL worker reports initialization, input preparation, sequence
generation, postprocessing and complete worker-request time separately:
`translator_initialization_seconds`, `translator_preparation_seconds`,
`translator_generation_seconds`, `translator_postprocessing_seconds`,
and `translator_worker_request_seconds`. Existing `translator_inference_seconds`
remains the end-to-end translation-function duration for backward comparison.
Older diagnostics used broader timers that included model loading; do not treat
those historical values as pure generation latency.

The transcript language-span lexicon includes common spoken product/design
terms (for example, team, college, dashboard, feedback, wireframes, navigation,
features and prototype). These are conservative lexical hints, not a general
language-identification model; unknown terms may remain `und`.

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
IndicTrans2 keeps `use_cache=False`; this is separate from a bounded
application-level cache of completed translations. Identical text/language/beam
requests can reuse the prior completed result without running the model again.

The WSL provider defaults to four-beam search. For a controlled CPU speed test,
set the optional beam count before starting Streamlit:

```powershell
$env:SPEECHMIND_TRANSLATION_BEAMS = "2"
.venv\Scripts\python.exe -m streamlit run app.py
```

Allowed values are `1`, `2`, and `4`; the default remains `4`. Lower beam
counts may change translation quality. Compare the meaning on the same
recording and return to four beams if important wording or negation changes.
The active beam count and result-cache status are shown in diagnostics.

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


## Audio quality and transcript presentation

SpeechMind preserves pauses and quiet speech during decode and reports duration,
peak/RMS level, clipped-sample share and near-silence share. These are signal
measurements, not a speech/noise classifier or correctness probability. No
denoising, trimming or automatic gain change is applied.

The final app displays the untouched ASR transcript separately from the English
meaning and exports the complete analysis as JSON. Transcript editing is not
part of the final user-facing workflow.

## Optional Romanized-Hindi transliteration

The code contains an optional local AI4Bharat IndicXlit adapter. It converts
only transcript tokens already classified as Romanized Hindi; English and
unclassified words remain unchanged, and the visible transcript is never
rewritten.

**Do not install `requirements_hinglish_transliteration.txt` into the primary
Windows `.venv`.** The published `ai4bharat-transliteration==1.1.3` package
pulls the legacy `fairseq` build dependency. The current Windows installation
attempt failed while building Fairseq, and the upstream project has a long-lived
Windows installation issue. The package also depends on older OmegaConf
metadata that current pip rejects. This path is therefore optional and is not
currently enabled on the Windows development environment.

When the dependency is absent or fails, SpeechMind keeps the original
transcript and reports the English-meaning step as unavailable for that
Romanized-Hindi route. No hosted API is used. Do not change the main
requirements or downgrade the primary environment's pip to work around this.
