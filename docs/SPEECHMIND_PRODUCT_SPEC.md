# SpeechMind — Canonical Product Specification

**Status:** Product contract. Treat this document as the stable reference for product purpose, user experience, scope, and output semantics. Implementation details and model choices may evolve; the principles in this document must not be silently changed.

## 1. Product identity

SpeechMind is an automatic multilingual, code-switching speech-understanding system designed around how people in India naturally speak.

Its primary audience is people who speak English, Hindi, Hinglish, and selected major Indian languages. It must accept natural speech without asking the speaker to identify the language first.

SpeechMind is not another generic voice-to-text application. Its distinguishing job is to:

1. Recognize natural Indian speech, including mid-sentence language switching.
2. Preserve what the speaker actually said in an original-language transcript.
3. express the transcript's meaning naturally in English, without replacing the original transcript.
4. Analyze the linguistic meaning and the vocal delivery independently.
5. Combine both signals into a careful interpretation, while explicitly showing uncertainty and disagreements.

## 2. Product objective

**Make naturally spoken Indian multilingual and code-mixed audio understandable in English, while explaining both what was said and how it was vocally expressed.**

The central principle is:

> SpeechMind adapts to the speaker. The speaker should not have to adapt to SpeechMind.

## 3. Intended language scope

The initial priority is deliberately focused:

- Core: English, Hindi, and Hindi–English code-switching (Hinglish).
- Initial regional expansion: Bengali, Marathi, Tamil, Telugu, and Punjabi.

This is a product prioritization, not a claim that every current model supports every language equally. A language is only considered supported in a release after representative evaluation.

Do not add a language selector to the normal recording flow. Do not ask users to choose "Hindi", "English", or "Hinglish". Language identification and code-switch handling are automatic system responsibilities.

"Hinglish" is a speaking pattern, not a user-selected language. A recording may switch languages several times within one sentence. Romanized Hindi, Devanagari Hindi, English, and mixed-script output all need consideration.

## 4. Non-negotiable user-experience principles

1. **No language selection before recording.** The user uploads or records speech and presses Analyze.
2. **Preserve the source.** The original transcript is retained and shown separately from its English meaning.
3. **Do not force one script.** Preserve English words as English where the ASR model supports mixed-script output; do not transliterate everything into Devanagari merely for consistency.
4. **Do not force one source language.** A recording can contain multiple languages; a single file-level language label must not control the entire pipeline.
5. **Translation is not transcription.** Translation may clarify meaning, but must never overwrite or silently rewrite the original transcript.
6. **Analyze audio and text independently.** The voice branch receives the original audio. The text branch receives the original transcript (and, when appropriate, its English meaning as an additional view). Translation must not be used as a substitute for the original text.
7. **Show disagreement.** Positive words delivered with sadness, or worried words delivered calmly, are meaningful differences between channels. Do not conceal them behind a single weighted label.
8. **Allow uncertainty.** The system may say that a result is inconclusive or that a branch is unavailable. It must not invent a confident answer.
9. **Separate quality from correctness.** A model score, token-quality heuristic, or softmax output is not an accuracy percentage unless calibrated and validated as such.
10. **No medical diagnosis.** SpeechMind describes linguistic and vocal expression. It does not diagnose anxiety, depression, stress disorders, or other medical or psychological conditions.

## 5. Canonical processing stages

### Stage A — Audio ingestion and preservation
Accept a short audio recording or upload. Decode to a standard waveform for model inputs, but retain the unmodified decoded waveform for the voice branch. Use voice activity detection and conservative chunking for long audio. Do not remove pauses or quiet sections in a way that destroys emotional or prosodic evidence.

### Stage B — Automatic multilingual/code-switch ASR
Transcribe without a user language choice. Prefer models trained for Indian accents and code-switching. The ASR layer should return:
- Original transcript.
- Available language / code-switch metadata, preferably at word or segment level.
- Timestamps when available.
- Model/provider identity.
- Available recognition-quality diagnostics, clearly marked as calibrated or uncalibrated.

### Stage C — Original transcript
Keep the best source-language rendering as its own artifact. Do not replace it with an English translation, normalize away Hindi/English switching, or rewrite words based on semantic guesses. Any optional normalization must be a separate display/export view.

### Stage D — English meaning
Translate the complete code-mixed transcript directly into natural English. The translation should preserve intent, names, numbers, negation, tense, and conversational nuance. It must not summarize or add information. The original transcript remains unchanged.

### Stage E — Parallel understanding
Run two independent branches:
- **Text understanding:** emotion and sentiment communicated by the words. It must be trained/evaluated on Hindi, English, and code-mixed or romanized Hinglish, not only monolingual English comments.
- **Voice understanding:** vocal emotional expression and acoustic/prosodic characteristics from the original waveform. It must not infer emotion solely from transcript meaning.

### Stage F — Reliability-aware fusion
Fuse the branch outputs using measured branch reliability, transcription quality, and validation/calibration evidence. Fixed weights may be used only as a temporary baseline and must be labeled provisional. Preserve each branch result and identify agreement or disagreement. The fusion layer must be able to abstain or return an inconclusive result.

### Stage G — Final result
Show:
- Original transcript.
- Meaning in English.
- Automatically detected language(s)/code-switch indication, when supported.
- Textual emotion / sentiment.
- Vocal emotion / prosodic indicators.
- Combined interpretation, including disagreement when relevant.
- Separate recognition/model-quality indicators and explanatory caveats.

## 6. Emotion and interpretation semantics

The initial shared emotion vocabulary is joy, sadness, anger, fear, surprise, disgust, and neutral. The model taxonomy can evolve only with a documented mapping and evaluation plan.

Prefer distributions or multi-label evidence over pretending every utterance has exactly one emotion. "Neutral" must not become a default destination for unknown or unmapped labels.

Text and voice are different kinds of evidence:
- Text expresses what the words communicate.
- Voice expresses how the speaker delivered them.
- The combined interpretation explains how these signals relate.

Do not infer a person's actual internal state from either channel.

## 7. Evaluation requirements

Each component must be evaluated separately, and then as a whole.

- ASR: WER and CER, normalized carefully for script; report separately for English, Hindi, Hinglish, and each regional language. For code-switching, evaluate language-switch/token behavior where references permit.
- Translation: human evaluation of meaning preservation, especially negation, tense, numbers, named entities, and code-mixed idioms.
- Text emotion: macro-F1, per-class precision/recall, and results separated by language/script.
- Voice emotion: speaker-disjoint macro-F1, per-class metrics, and results separated by recording/domain where possible.
- Fusion: compare with each individual branch, report agreement/disagreement cases, and assess calibration/abstention.
- Product: latency, memory, failure rate, and privacy/backend behavior.

Never compare model scores from non-equivalent datasets as if they were a controlled head-to-head result.

## 8. Stable product contract vs replaceable implementation

The product objective, automatic language handling, separate original transcript and English meaning, independent text/voice analysis, disagreement-aware fusion, and non-diagnostic framing are stable.

ASR provider, translation provider, model backbone, datasets, fine-tuning method, fusion algorithm, UI arrangement, hosting, and local/cloud deployment may change as evidence improves. Any implementation change must preserve the stable contract above and document its trade-offs.
