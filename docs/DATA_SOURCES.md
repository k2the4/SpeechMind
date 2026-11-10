# SpeechMind — Data Sources and Acquisition Plan

This file is the working list of datasets and services for the current Hinglish-first development plan. Review the source license and access conditions before redistribution or use outside this project.

## 1. ASR and code-switching data

### IITG-HingCoS — priority human-transcribed Hinglish benchmark

Official page: https://www.iitg.ac.in/eee/emstlab/HingCoS_Database/HingCoS.html

- 25 hours of matching Hindi-English code-switched speech.
- 9,251 code-switched spoken utterances and corresponding text.
- Access is requested through the dataset page; this is not an unrestricted one-click download.
- Use it first to build a small human-reference ASR evaluation manifest.
- Keep its train/test or speaker split rules intact. Do not create leakage by putting utterances from the same speaker in training and test.

### IndicVoices — natural Indian speech

Dataset: https://huggingface.co/datasets/ai4bharat/IndicVoices

- Large natural-speech collection across 22 Indian languages and many Indian districts/speakers.
- Hugging Face access is gated; the user must accept the dataset conditions.
- The full collection is hundreds of gigabytes. Do not download the full corpus to the 16 GB PC.
- Use a small approved split/configuration for language coverage and ASR validation, respecting the dataset's terms.
- This collection is not an emotion-labelled corpus.

## 2. Text emotion

### HinEmo-9 — initial Hinglish text-emotion baseline

Repository: https://github.com/im-rakeshsahani/hinemo9  
Paper / DOI: https://doi.org/10.5281/zenodo.22041924  
Dataset license: CC BY 4.0 for the authors' annotations and curation.

The SpeechMind downloader saves only the dataset files and documentation needed for the baseline:
- `data/train.jsonl`
- `data/dev.jsonl`
- `data/test.jsonl`
- `data/gold.jsonl`
- `data/checksums.txt`

Important distinctions from the dataset card:
- 30,436 English, Devanagari Hindi and romanized Hinglish YouTube comments.
- 3,363 human-gold items were independently annotated by three annotators.
- 27,073 additional records are model-derived with human review; their reported agreement with gold is 78.4% overall and much lower for some classes.
- Dev/test are human-gold. The training script explicitly requires that held-out evaluation data remains human-gold.
- The dataset has nine labels: Love, Joy, Anger, Sadness, Fear, Surprise, Nostalgia, Devotion and Neutral.
- The source domain is YouTube comments, not spoken conversations.
- The CC BY 4.0 license covers annotations/curation, not underlying third-party YouTube comment text. Do not redistribute the underlying text; review platform terms and the full dataset card.

SpeechMind's training script down-weights model-derived examples by the dataset's reported per-class agreement. Use results as a text baseline only. Do not present its benchmark as real speech-emotion accuracy.

### EmoInHindi — possible additional resource

Resource page: https://www.iitp.ac.in/~ai-nlp-ml/resources.html

A Hindi dialogue emotion resource. Review the accompanying paper, licensing and annotation structure before adapting it. Multi-label annotations must not be silently converted to single-label data.

### GoEmotions — possible English supplement

Dataset: https://huggingface.co/datasets/mrm8488/goemotions

English Reddit comments with 27 emotion labels plus neutral. It is multi-label and differs from conversational speech. Use only after deliberate label mapping and license/domain review.

## 3. Transcript language identification

### AI4Bharat IndicLID

Repository: https://github.com/AI4Bharat/IndicLID  
License: MIT for the code and models.

The included download script installs the native-script and romanized fastText models from the v1.0 GitHub release. IndicLID predicts language/script for transcript windows; it does not recover language boundaries directly from raw audio.

Its Roman-script benchmark is materially weaker than its native-script benchmark. SpeechMind reports the model and its per-window outputs so uncertainty is visible.

## 4. Voice emotion

### Existing acted-speech baseline

- RAVDESS: existing local data, 1,440 recordings used by the SpeechMind baseline.
- CREMA-D: existing local data, 7,442 recordings used by the SpeechMind baseline.

These datasets provide an initial acted-English vocal-emotion baseline, not spontaneous Indian-language emotional speech.

### Future natural Indian voice-emotion collection

We need audio with explicit consent and human annotation of vocal expression. Speech transcripts alone do not provide vocal emotion labels; acted-audio labels must not be reused as text-emotion labels. A future collection should define annotation dimensions, neutral examples, annotator agreement and speaker-disjoint evaluation before training.

## 5. Optional ASR and translation service

Bodhan API console: https://console.bodhan.ai/  
API docs: https://console.bodhan.ai/api-docs/  
Speech model: https://huggingface.co/bodhan-ai/indic-transcribe-flex  
Translation model: https://huggingface.co/bodhan-ai/indic-translate

The current adapter uses:
- `POST /v1/audio/transcriptions` with model `indic-transcribe`, omitting a language parameter so the service can handle language detection/code-switching automatically.
- `POST /translate` with `{"text":"...","target_language":"en"}`; source-language is inferred.

Current published price pages list ₹0.10 per audio minute for ASR and ₹0.20 per 10,000 output tokens for translation. Verify current prices and provider retention/processing terms before use.

Audio is only sent to the hosted service when the local `BODHAN_API_KEY` is configured. No key means local Whisper fallback.

## 6. Acquisition order

1. Configure optional Bodhan key only if cloud processing is acceptable.
2. Download IndicLID model weights.
3. Request IITG-HingCoS access and build the ASR reference manifest.
4. Download HinEmo-9 and train the text baseline.
5. Evaluate on real SpeechMind transcripts and a human-reviewed Hinglish sample.
6. Investigate IndicVoices selectively for language and speaker diversity.
7. Build a consented natural Indian voice-emotion dataset before claiming regional voice-emotion support.
