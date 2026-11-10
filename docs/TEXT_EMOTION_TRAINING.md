# HinEmo-9 text-emotion baseline

The primary first text-emotion model for SpeechMind is a CPU-friendly classifier trained on HinEmo-9. It is a research baseline for English, Devanagari Hindi and Romanized Hinglish text.

## Dataset structure and label provenance

Source: https://github.com/im-rakeshsahani/hinemo9

The dataset card reports:
- 30,436 YouTube comments in English, Devanagari Hindi and Romanized Hinglish.
- 3,363 human-gold samples independently labeled by three annotators.
- 27,073 model-derived samples with reported 78.4% agreement to gold; agreement varies substantially by emotion (including lower values for Joy and Love).
- The training split contains both sources. Dev/test are human-gold only.
- Nine labels: Love, Joy, Anger, Sadness, Fear, Surprise, Nostalgia, Devotion and Neutral.

The authors' CC BY 4.0 license covers annotations and curation, not underlying third-party YouTube text. The downloader is for local research/training only. Do not redistribute the downloaded comment text. Review the source dataset card and platform terms.

## Download

Run from the SpeechMind project root:

```powershell
.\DOWNLOAD_HINEMO9_DATA.bat
```

It installs the selected files under `data/hinemo9/data`:
- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`
- `gold.jsonl`
- `checksums.txt`

## Training method

Run:

```powershell
.\TRAIN_HINEMO9_TEXT_EMOTION.bat
```

The training script uses character n-grams (2–6) and word n-grams (1–2), TF-IDF and a class-balanced LinearSVC. This is intentionally CPU-friendly and character features help with spelling variation in Romanized Hindi.

Human-gold training records have weight 1.0. Model-derived records receive a lower class-specific weight based on the dataset's reported agreement. Dev/test reject non-gold examples. The dev set selects the regularization parameter; the test set is used once for final reporting.

## Outputs

- Checkpoint: `checkpoints/text_emotion_hinemo9`
- Metadata: `checkpoints/text_emotion_hinemo9/speechmind_metadata.json`
- Metrics: `training_outputs/hinemo9_text_emotion/test_metrics.json`

The report includes overall and per-language accuracy/macro-F1, per-class precision/recall/F1 and a confusion matrix.

## Interpretation and limits

- This is text emotion, not voice emotion.
- YouTube comments are not natural speech transcripts.
- Model-derived training labels are noisy and especially weak for some emotions.
- The published gold test set is useful for a baseline but has known class-composition and domain limitations.
- Validate on a separate human-labeled set of SpeechMind-generated speech transcripts before claiming spoken Hinglish text-emotion accuracy.
- Its nine labels differ from the current seven-label WavLM voice model. Fusion preserves text-only Love, Nostalgia and Devotion; it does not invent those emotions in the voice branch.
- LinearSVC decision scores converted to a softmax-like distribution are uncalibrated. They are not probabilities of correctness.
