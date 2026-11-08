# SpeechMind: model and data readiness

## Current inference baseline

- ASR: Whisper-small for automatic language detection/English; IndicConformer-600M for supported Indic languages.
- Voice emotion: `Dpngtm/wav2vec2-emotion-recognition`, a seven-class English model.
- Text emotion: intentionally disabled until a fine-tuned MuRIL checkpoint is present.
- Fusion: the available branch is used; scores are explicitly uncalibrated.

## Important dataset leakage warning

The configured voice-emotion model card states that it was fine-tuned on TESS, CREMA-D, SAVEE and RAVDESS, and describes the model as English-language. It also reports its own validation figures, but SpeechMind has not independently reproduced those results. Therefore:

- Do not fine-tune this checkpoint on the same RAVDESS files and then report those files as an independent test set.
- Do not present its Hindi or other Indian-language predictions as validated. They remain experimental until evaluated on Indian-language speech.
- RAVDESS remains useful for checking filename parsing and for an English ASR smoke/word-error-rate evaluation, with the usual possibility of pretraining-data overlap considered.

## Candidate Indian-language emotion data

### BhavVani

The BhavVani work was presented at Interspeech 2024 and describes a Hindi speech-emotion corpus. The Hugging Face dataset card lists Hindi/English metadata and 7,991 rows; it explicitly states that audio files are provided by request and that use is limited to non-commercial research. Use it only after the access request is approved and retain the required attribution. It is the preferred candidate for adapting and evaluating the voice branch.

### IITKGP-SEHSC / IITKGP-SESC

These are Indian-language emotional-speech corpora (Hindi and Telugu respectively). Their dataset pages indicate request-based access. They are suitable candidates if the project obtains permission and the corpus terms are confirmed.

### BHAAV

BHAAV contains 20,304 Hindi sentences labeled with five categories: anger, joy, suspense, sad and neutral. The official repository says the corpus is for research use and requires citation; commercial use requires contacting the maintainers. It is a text corpus derived from stories, not a corpus of spontaneous speech transcripts. It can be used for an initial Hindi text-emotion baseline, but that domain mismatch must be reported and it should not be represented as final validation for transcribed conversation.

## Training and evaluation protocol

1. Keep a speaker-independent train/validation/test split for speech-emotion data. Never split adjacent recordings from the same speaker across train and test.
2. Train/adapt the voice branch on permitted Indian-language audio. Keep an untouched speaker-disjoint test set.
3. Train the text branch on Hindi text with documented label mapping. Keep story/source separation for BHAAV to reduce leakage across related sentences.
4. Report per-class precision, recall, macro-F1 and confusion matrices, not accuracy alone.
5. Compare voice-only, text-only and fused predictions on the same eligible utterances.
6. Calibrate confidence on validation data only; do not describe maximum softmax/fusion score as probability of correctness.
7. Report results separately by language and dataset. A model evaluated on RAVDESS is not thereby validated for Hindi speech.

## Sources

- Voice model card: https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition
- BhavVani dataset card and access conditions: https://huggingface.co/datasets/ag2003/bhavvani
- BhavVani Interspeech 2024 paper: https://www.isca-archive.org/interspeech_2024/goel24_interspeech.pdf
- IITKGP dataset information: https://cse.iitkgp.ac.in/~ksrao/res.html
- BHAAV official dataset page: https://github.com/midas-research/bhaav
- BHAAV Zenodo record: https://zenodo.org/records/3457467
- RAVDESS official description and filename labels: https://zenodo.org/records/1188976
