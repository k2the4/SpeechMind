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

SpeechMind is an experimental multilingual speech-understanding application for natural English, Hindi and Hinglish speech.

It keeps the original transcript, English meaning, semantic text analysis and vocal analysis as separate outputs. It does not diagnose medical or psychological conditions.

## Hosted deployment

See [docs/DEPLOY_HUGGINGFACE_SPACES.md](docs/DEPLOY_HUGGINGFACE_SPACES.md) for Docker Space setup, private checkpoint configuration and deployment instructions.

The hosted CPU configuration may take around a minute or more to process short Hinglish recordings. Model predictions are experimental and should not be treated as verified emotional states. Do not upload private or sensitive recordings to a public demo.
