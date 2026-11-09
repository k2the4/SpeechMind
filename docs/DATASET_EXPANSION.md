# SpeechMind dataset expansion

## Why add CREMA-D?

The current experiment uses RAVDESS speech: 1,440 clips, 24 actors, seven mapped emotions. CREMA-D adds approximately 7,442 acted English speech clips from 91 actors and six emotions: anger, disgust, fear, happy, neutral and sad.

The training pipeline maps CREMA-D labels into SpeechMind's seven-label schema:
- HAP -> joy
- SAD -> sadness
- ANG -> anger
- FEA -> fear
- DIS -> disgust
- NEU -> neutral

CREMA-D does not contain a surprise class. Surprise remains represented by RAVDESS. The evaluation report includes combined held-out results and separate CREMA-D/RAVDESS metrics so a good result on one corpus cannot hide a weak result on the other.

## Obtain CREMA-D

The official CREMA-D repository asks users to complete its access form and uses Git LFS for audio/video assets. Read the repository's access notes and license before downloading:
- Official repository: https://github.com/CheyneyComputerScience/CREMA-D
- Access/use registration: linked in the official repository README
- License: Open Database License (ODbL) plus Database Contents License for individual contents.

For SpeechMind, only the audio WAV folder is required. Install Git LFS for Windows first:
https://git-lfs.com/

Then run `DOWNLOAD_CREMA_D.bat` from the SpeechMind root. It clones repository metadata with LFS smudging disabled, checks out only `AudioWAV`, pulls only WAV LFS objects in that folder, and copies the audio to:

```
data/crema_d/AudioWAV/
```

This avoids downloading the video folder. Ensure approximately 2 GB or more of free disk space.

Do not commit dataset files, WAVs, model checkpoints, or caches to GitHub.

## New training representation

The updated pipeline uses:
1. the last four WavLM hidden-state layers averaged together;
2. global mean and standard deviation;
3. 10th and 90th percentile pooling.

This creates 3,072-dimensional representations for WavLM-base-plus. The same helper is used for training and inference to prevent a train/inference feature mismatch.

Embeddings are cached in:
`training_outputs/wavlm_multicorpus/wavlm_features_cache.npz`

The cache fingerprint includes file size/mtime, corpus, label, speaker split, model and preprocessing version. Re-running classifier experiments does not require feature extraction if the inputs are unchanged.

## Evaluation design

- Each corpus is split by speaker; no speaker appears in more than one split.
- SVM selection uses validation macro-F1.
- Corpus-balanced sample weights prevent the larger CREMA-D corpus from dominating solely by clip count.
- Final test metrics are reported combined and per dataset.
- The test split is not used to select the classifier.

## Training outputs

- New checkpoint: `checkpoints/voice_emotion_multicorpus_cpu/`
- Metrics: `training_outputs/wavlm_multicorpus/test_metrics.json`
- Manifest and split report: `training_outputs/wavlm_multicorpus/`
- Feature cache: `training_outputs/wavlm_multicorpus/wavlm_features_cache.npz`

The previous CPU WavLM/SVM checkpoint at `checkpoints/voice_emotion_wavlm_cpu/` is preserved. The app prefers the new multi-corpus checkpoint only when its metadata and feature dimensions validate.

## Future Indian-language data

CREMA-D expands acted English speech; it does not solve Indian-language emotion recognition. The Hindi catalog lists:
- Vishal B. Hindi Speech Emotion Recognition dataset: 3,200 acted WAV samples, open download, but its license is not stated in the catalog. Do not use it for redistributed/commercial use until the rights holder's terms are confirmed.
- Bansal & Dev Emotional Hindi Speech Corpus: 6,048 samples from 28 speakers, access on request.
- IITKGP-SEHSC: 12,000 samples, 10 speakers, eight emotions, access on request; request permission and usage terms before integrating.

Once we obtain a dataset with clear permission, it should be added as a distinct corpus with speaker-disjoint splits and per-corpus test reporting. English and Hindi metrics must remain separate.
