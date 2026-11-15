---
title: SpeechMind
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# SpeechMind

SpeechMind is an experimental multilingual speech-understanding application for natural English, Hindi, Hindi-English (Hinglish), Bengali, Tamil, Telugu and Marathi speech.

It identifies the spoken language, preserves the original transcript, produces an English meaning, analyses emotion from the original words and voice in parallel, and presents a confidence-aware interpretation. Feature availability is checkpoint-dependent: a text-emotion model is used only for languages it explicitly declares as supported. Unsupported analysis is reported as unavailable rather than guessed. SpeechMind is descriptive and does not diagnose medical or psychological conditions.

See [docs/ASR_AND_EMOTION_ROUTING.md](docs/ASR_AND_EMOTION_ROUTING.md) for current model routing, uncertainty handling, evaluation policy and hosted-ASR considerations.

For multi-recording production evaluation, see [docs/MULTI_RECORDING_EVALUATION.md](docs/MULTI_RECORDING_EVALUATION.md) and run `EVALUATE_SPEECHMIND_HIACC.bat` after preparing the HiACC manifest.

For a controlled same-audio comparison of the ASR candidates, see [docs/PAIRED_ASR_BENCHMARK.md](docs/PAIRED_ASR_BENCHMARK.md) and run `COMPARE_ASR_MODELS_HIACC.bat` after completing a full-pipeline evaluation.


## Hosted deployment

See [docs/DEPLOY_HUGGINGFACE_SPACES.md](docs/DEPLOY_HUGGINGFACE_SPACES.md) for Docker Space setup, private checkpoint configuration and deployment instructions.

The hosted CPU configuration may take around a minute or more to process short Hinglish recordings. Model predictions are experimental and should not be treated as verified emotional states. Do not upload private or sensitive recordings to a public demo.

Audio-quality diagnostics and complete JSON result export are part of the app workflow. Automatic Whisper vocabulary scouting is opt-in, not enabled by default. Optional local Romanized-Hindi transliteration is documented in the routing guide.


## Conversation intelligence

SpeechMind can use the Gemini API free tier (`gemini-2.5-flash-lite`) without downloading model weights. Set `GEMINI_API_KEY` before launching the app. Free-tier content may be used to improve Google products; do not send sensitive customer recordings. Ollama remains an optional offline alternative. Without either provider, deterministic analysis is used. See [docs/LOCAL_LLM_SETUP.md](docs/LOCAL_LLM_SETUP.md).
