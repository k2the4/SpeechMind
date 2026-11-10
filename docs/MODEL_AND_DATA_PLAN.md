# SpeechMind — Model, Dataset and Experiment Plan

## A. Recommended first ASR experiment

Compare these on exactly the same evaluation recordings:

1. Current local Whisper Small, automatic transcription (baseline).
2. Bodhan Indic-Transcribe API (explicit opt-in hosted candidate).
3. Srota union model (`moorlee/qwen3-asr-0.6b-hinglish`) for mixed-domain Hinglish.
4. Srota-Conv (`moorlee/qwen3-asr-0.6b-hinglish-hiacc-v1`) for spontaneous/conversational Hinglish.
5. Indic-Transcribe Core/Flex model checkpoints if a suitable GPU environment is available.

Srota-Conv's model card reports 14.23% WER on its HiACC conversational test and notes a speaker-overlap caveat for that split. Srota is the union candidate for conversational and tutorial speech. Treat these as model-card-reported, in-domain claims, not independent proof of SpeechMind performance. Our own test must keep speakers unseen and must not train on the test transcript/audio.

Indic-Transcribe is designed for 25+ Indian languages and code-switching. Core emphasizes native-script output; Flex provides native, mixed-script, and Romanized rendering options. It is a large 1.2B model, so use the hosted API or a separate GPU environment for initial comparison rather than assuming the GT 730 can run it.

## B. First datasets

### 1. HiACC — initial Hinglish benchmark
- 5.24 hours; 5,176 utterances; adult and child speakers.
- Read and spontaneous speech.
- 16 kHz mono WAV with aligned transcripts and code-switch annotations.
- Official record: https://zenodo.org/records/15551669
- Official paper's data-access section states CC BY-NC 4.0 / academic-research use. A later third-party model card describes the corpus as CC BY 4.0, which conflicts with the original data paper. Follow the original record/paper terms unless the data authors clarify otherwise; do not use it commercially or redistribute it on an assumption.
- The corpus includes children. Keep it for research evaluation only, restrict access, and do not publish recordings or identifying metadata.

Download via `DOWNLOAD_HIACC.bat`. The script queries Zenodo metadata and downloads the dataset archive(s); it does not put dataset files in Git.

### 2. OpenSLR-104 — tutorial/technical Hinglish
- Hindi-English train: approximately 89.86 hours.
- Hindi-English test: approximately 5.18 hours.
- 16 kHz audio with transcripts.
- Official: https://www.openslr.org/104/
- Official OpenSLR page lists CC BY-SA 4.0. Preserve attribution and share-alike obligations; obtain a legal/academic review before any commercial redistribution or derivative release.
- The distribution is spoken tutorials, not ordinary casual conversation. Use it as a separate technical/tutorial domain benchmark, not as a substitute for conversational Hinglish.

The full training archive is several GB. Do not download until the HiACC pipeline is working. The official test archive is much smaller and can be used for a separate benchmark.

### 3. IndicVoices — later regional-language expansion
- Natural/read/extempore/conversational speech across 22 Indian languages and 51K speakers.
- Hugging Face dataset: https://huggingface.co/datasets/ai4bharat/IndicVoices
- License listed as CC BY 4.0, but access requires accepting the dataset terms and sharing contact information. The total archive is extremely large (the dataset card lists 745 GB); use only selected language/split subsets after access approval.
- Planned languages: Bengali, Marathi, Tamil, Telugu, Punjabi, alongside Hindi. Select subsets and splits based on the actual evaluation need rather than downloading the full release.

### 4. IITG-HingCoS — not first priority
- Approximately 25 hours of Hindi-English code-switching speech.
- The official request page currently lists paid delivery (INR 50,000 + 18% GST for an Indian organization) and research/development-only terms; it prohibits commercial benefit from the dataset or derivatives.
- Official request page: https://www.iitg.ac.in/eee/emstlab/HingCoS_Database/db_copy.php
- Do not request/purchase it for the first iteration.

## C. Text emotion data

### HinEmo-9
- 30,436 English, Devanagari Hindi, and Romanized Hinglish YouTube comments.
- 3,363 human-gold examples are independently annotated by three annotators; keep this tier for validation/testing.
- 27,073 model-derived labels have lower agreement with gold and are noisy training material only.
- Repository: https://github.com/im-rakeshsahani/hinemo9
- Dataset record: https://zenodo.org/records/22041925
- Review the dataset card, license, label schema, and annotation guideline before using it. Do not automatically squeeze all nine categories into SpeechMind's seven core labels. Keep non-equivalent labels out of the seven-class training set until an annotation mapping is agreed.

Text comments are not spoken conversations. Use them as an initial text model resource, then collect/annotate consented real spoken transcripts for domain evaluation.

## D. Voice emotion data and model comparison

Keep the current RAVDESS + CREMA-D WavLM/SVM as baseline. Benchmark emotion2vec+ base as an alternative; it is approximately 90M parameters and exposes angry, disgusted, fearful, happy, neutral, other, sad, surprised, and unknown outputs. Do not map other/unknown into neutral.

RAVDESS/CREMA-D are acted speech. Add natural Indian speech only where labels are human-annotated and the dataset's license/consent permits emotion research. Language/audio corpora without emotion annotations can help ASR, but cannot become ground-truth emotion data.

## E. Evaluation protocol

Every ASR row in a manifest should include:
- audio path;
- exact reference transcript;
- dataset;
- official split;
- speaker ID, when supplied;
- language/domain tag.

Never alter official test references to fit a candidate's output. Use clearly documented normalization for WER/CER and keep both normalized and original transcripts available.

For emotion tasks, preserve annotator disagreement if available. Report macro-F1 and per-class metrics, plus results by language, script, corpus, and speaker cohort. Do not merge audio-level and text-level labels as if they were the same annotation.

## F. Hardware and workflow

Current development machine: Windows, Ryzen 5 3500, 16 GB RAM, GT 730; PyTorch is CPU-only. It is suited to data preparation, lightweight SVM fitting, unit testing, and app integration. Large ASR/full-transformer fine-tuning should use a separate GPU/Colab/Kaggle setup.

Experimental model packages (for example `qwen-asr` or FunASR) should use a separate environment or pinned optional requirements. Do not upgrade the working app's Transformers/PyTorch stack just to try a candidate.
