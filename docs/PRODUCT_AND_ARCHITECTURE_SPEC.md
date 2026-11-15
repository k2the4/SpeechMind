# SpeechMind — Product and Architecture Specification

**Status:** Fixed product direction. Technical implementations may evolve without changing the objective.
**Canonical reference:** This document governs product scope, user experience, component responsibilities, and processing order.

## 1. Product objective

SpeechMind is a multilingual **customer-support conversation intelligence** platform. It turns recorded support calls into evidence-backed, reviewable operational insights for support agents, quality reviewers, and team leads.

The product is not merely speech-to-text and does not replace a human quality analyst. It must:
1. Transcribe the conversation faithfully, preserving Hindi, English, Hinglish, and natural code-switching.
2. Produce an English meaning separately without mutating the original transcript.
3. identify the customer-support issue and relevant operational topic(s), with evidence spans or quoted transcript cues.
4. Surface review signals such as a possibly unresolved issue, repeat contact, requested follow-up, escalation language, customer-reported impact, and agent commitments.
5. Produce a concise case brief and structured next-step suggestions that are traceable to the conversation.
6. Make uncertainty, missing audio/transcript evidence, unsupported language coverage, and model limitations visible.
7. Aggregate reviewed call records into team-level trends only when records have been processed and stored with appropriate privacy controls.

The initial target workflow is post-call review of customer-support conversations. Live agent assistance, automatic CRM updates, customer identity resolution, and autonomous case closure are out of scope for the MVP.

SpeechMind must distinguish transcript evidence from model interpretation. It must not claim that a complaint is verified, that an issue is actually resolved, that a customer is objectively angry, or that an agent made a commitment unless the recording supports that statement. Vocal emotion is an experimental acoustic cue, not a reliable measure of a person's internal state.

SpeechMind is not a medical, psychological, lie-detection, personality, or intent-diagnosis product.

## 2. Intended language scope

Core:
- English
- Hindi
- Hinglish (Hindi-English code-switching, including Romanized Hindi and mixed scripts)

Planned regional-language expansion is deliberately limited. Initial candidates are Bengali, Marathi, Tamil, Telugu, and Punjabi. A language must be evaluated before it is advertised as supported.

Language selection is not a required input. The system should detect language and language switching from audio. If detection is uncertain, it should preserve uncertainty rather than silently force a language.

## 3. User-facing experience

1. Upload a customer-support call recording or record a short sample.
2. Press Analyze; no manual language selection is required.
3. Review call metadata and audio-quality warnings.
4. Read the original transcript and English meaning side by side, with transcript evidence kept available.
5. Review the detected support topic(s), issue description, evidence cues, and confidence/uncertainty labels.
6. Review service signals: possible unresolved issue, repeat contact, follow-up request, escalation cue, customer impact, and agent commitment. A signal is a review prompt, not a verified outcome.
7. Read an extractive or model-generated case brief only when its method is clearly labelled. Extractive text must not be marketed as an AI summary.
8. Review suggested next steps, each linked to supporting transcript evidence and labelled as a suggestion.
9. Export a structured case report (JSON in the MVP).
10. Human reviewers can correct or confirm findings in a later workflow; the MVP must not silently treat heuristic output as ground truth.

The original transcript must never be overwritten by translation, transliteration, cleanup, or summarization. Preserve the audio and transcript as the source of truth. Keep original transcript, English meaning, detected topics, evidence cues, service signals, and recommendations as distinct fields.

## 4. Target processing architecture

Audio input
-> Decode and audio-quality diagnostics
-> Automatic ASR and language/code-switch identification
-> Immutable original transcript + timestamps/language spans + ASR diagnostics
-> English meaning generation (separate, with route/method disclosure)
-> Parallel analysis:
   A. Support issue/topic detection and transcript evidence extraction
   B. Service-signal detection (repeat contact, follow-up, escalation, unresolved wording, commitments)
   C. Text/semantic cues from original transcript
   D. Optional vocal/acoustic cues from audio
-> Evidence and uncertainty validation
-> Case brief + suggested next actions
-> Human-reviewable case report + JSON export
-> (Later phase) privacy-controlled multi-call aggregation and team dashboard

Conversation intelligence must not depend on an emotion label. Topic and service-signal outputs must remain available when emotion models are unsupported or unavailable. Each finding should carry its source evidence, method, and review status. Translation may assist analysis but must not be treated as ground truth when the translation route is uncertain.

## 5. Component responsibilities

### 5.1 Audio input and preparation
- Accept common audio uploads and in-app recording.
- Decode to consistent mono, 16 kHz audio for model compatibility.
- Preserve meaningful pauses and low-volume speech where possible.
- Compute quality indicators such as clipping, silence, duration, and noise where feasible.
- Do not apply destructive trimming or denoising without validating its effect on emotion recognition.

### 5.2 ASR and language identification
- Recognize speech automatically without a required language selector.
- Support code-switching within one utterance, especially Hindi-English/Hinglish.
- Preserve English and Indian-language words as spoken.
- Prefer segment/span-level language identification over assigning one language to an entire recording.
- Retain timestamps, decoder metadata, and honest quality indicators.
- Never describe an uncalibrated heuristic as probability that a transcript is correct.
- Use language-appropriate specialist candidates behind one automatic interface: IndicConformer is the current Hindi baseline; Whisper remains an English candidate; Srota is the leading Hinglish candidate identified for validation. Do not treat a single primary-language guess as sufficient for code-switched speech.
- Do not require a language selector in the user interface.
- Benchmark candidates on held-out data before production routing; do not repeatedly re-run a completed benchmark without a new model or hypothesis.

### 5.3 Original transcript
- Preserve actual spoken content and natural language mixture.
- Make script rendering a recognition/presentation decision, not a forced language selection.
- Keep transcript, language spans, timestamps, and ASR diagnostics as structured data.

### 5.4 English meaning
- Produce a natural English rendering of the complete meaning, including code-mixed input.
- Do not merely copy an English-heavy transcript and call it translation.
- Preserve names, numbers, intent, tense, negation, and important nuance.
- If translation is unavailable or uncertain, disclose that and retain the original transcript.
- Evaluate meaning preservation separately from ASR word error.

### 5.5 Text/semantic analysis
- Analyze the original transcript, including code-mixed and Romanized Hindi.
- Estimate linguistic/semantic emotion from words and phrasing.
- Use a shared, documented emotion taxonomy across branches.
- Treat MuRIL and XLM-R as candidates to evaluate; fine-tune with suitable human-labelled code-mixed data.
- Do not run an English-only model on Hindi/Hinglish and present it as multilingual.

### 5.6 Vocal/acoustic analysis
- Analyze audio independently of the transcript.
- Estimate vocal emotional expression using prosody, pitch, energy, rhythm, voice quality, and learned speech representations.
- Keep vocal emotion separate from semantic emotion.
- Estimate arousal from validated acoustic evidence or a separately validated model, not simply from the emotion label.
- WavLM remains the current baseline; emotion2vec is a candidate for controlled comparison.
- CREMA-D/RAVDESS results are baseline evidence only; they do not establish performance on spontaneous Indian speech.

### 5.7 Reliability-aware fusion
- Combine text and voice distributions only when each branch is available and valid.
- Account for ASR quality when weighting text analysis, but do not treat heuristic ASR scores as calibrated probabilities.
- Validate or learn fusion weights using validation data rather than assuming fixed 60/40 weights are universally correct.
- Preserve and display branch-level outputs.
- Detect and explain disagreement (for example, positive wording with subdued or distressed vocal delivery).
- Allow uncertainty or insufficient evidence; do not force a confident single label.
- Never claim that the output reveals the speaker's true internal emotional state.

## 6. Emotion taxonomy and output

Current shared taxonomy:
- joy
- sadness
- anger
- fear
- surprise
- disgust
- neutral

The taxonomy changes only through an explicit product decision and corresponding dataset/model migration. Keep these concepts distinct:
- Semantic/text emotion: what the words communicate.
- Vocal emotion: how the voice sounds.
- Valence: positive/negative/neutral dimension, where validated.
- Arousal: activation/intensity dimension, where validated.
- Combined interpretation: cautious synthesis of available evidence.

A top classifier score is not automatically a probability of correctness. Calibrated confidence is shown only after calibration is measured on representative held-out data.

## 7. Evaluation requirements

- ASR: WER, CER, language-span/code-switch handling, per-language and per-condition breakdowns.
- Translation: human-rated meaning preservation, adequacy, and critical error checks.
- Text emotion: macro-F1, per-class precision/recall/F1, confusion matrix, language/script breakdown.
- Voice emotion: speaker-independent macro-F1, per-class metrics, recording-condition breakdown.
- Fusion: held-out macro-F1, calibration, branch disagreement analysis, and abstention/uncertainty quality.

Avoid data leakage by keeping speakers disjoint across train/validation/test wherever identities are available. Keep natural, human-recorded test data separate from synthetic augmentation. Do not claim real-world Hinglish performance based only on acted English corpora.

## 8. Hardware and deployment constraints

The current development machine is Windows with 16 GB RAM and an NVIDIA GT 730. Treat modern model inference/training as CPU-first unless a compatible runtime is demonstrated. Large ASR/translation models may require hosted endpoints or remote GPU for benchmarking. Any hosted audio service requires explicit privacy, cost, retention, and consent review before adoption.

Training and evaluation must be reproducible through documented scripts and Windows batch launchers. Dataset downloads must be separate from training, include source/license/access notes, verify files, and never silently download huge datasets during application startup.

## 9. Current implementation status (baseline, not product contract)

The repository currently contains:
- Librosa-based audio decoding/resampling/normalization.
- Whisper Small language detection and transcription, with IndicConformer routing for a detected Indic language.
- IndicTrans2 distilled translation for supported Indic-language transcripts, isolated in a Linux/WSL worker on Windows.
- MuRIL text-emotion loading only when a fine-tuned checkpoint exists; an English-only fallback exists.
- WavLM Base Plus frozen embeddings and an SVM voice-emotion classifier trained on CREMA-D + RAVDESS.
- Rule-based late fusion with fixed heuristic weights.
- Streamlit upload/record interface.

These are implementation details that may change. They do not override the product objective. The language selector is not part of the intended product experience and must not return as a required step.

## 10. Development principles

1. Preserve the product objective and user journey.
2. Prefer evidence and controlled benchmarks over model popularity or assumptions.
3. Do not replace a working model until the replacement has been evaluated on comparable data.
4. Keep original transcript, English meaning, text analysis, voice analysis, and combined interpretation as distinct outputs.
5. Be transparent about unsupported languages, limitations, uncertainty, and external service use.
6. Keep training, data preparation, inference, and UI concerns modular.
7. Do not claim local execution, training, or testing unless it was actually performed.
8. Explain product-level changes before implementing them; routine engineering improvements can proceed within this specification.


## 11. Translation runtime on Windows

Windows SpeechMind keeps ASR and emotion inference in its existing environments.
IndicTrans2 runs in a separate WSL Linux virtual environment because the
IndicTransToolkit maintainers state that their package is not built or tested for
native Windows. The Windows application connects to a persistent JSONL worker;
the original transcript remains immutable and English meaning is a separate
result. Translation runs concurrently with semantic and vocal analysis after
ASR, so model loading/inference overlaps the other branches. See
`docs/TRANSLATION_WSL_SETUP.md`.
