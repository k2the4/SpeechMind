# Text-emotion branch setup

SpeechMind deliberately does not load base MuRIL as though it were an emotion
classifier. A classification head must be trained for the seven labels used by
the fusion layer. For English transcripts, the application can use the
pretrained English-only fallback `j-hartmann/emotion-english-distilroberta-base`.
That fallback does not support Hindi or other Indic languages.

## 1. Prepare a labeled CSV

Create `data/text_emotion/text_emotion.csv` with these columns:

```csv
text,emotion,language
"I am excited about the result",joy,en
"I am worried about tomorrow",fear,en
"मुझे यह सुनकर बहुत खुशी हुई",joy,hi
"मैं परीक्षा को लेकर चिंतित हूँ",fear,hi
```

Each row must have one primary, human-checked emotion label. Supported labels
are: `joy`, `sadness`, `anger`, `fear`, `surprise`, `disgust`, and
`neutral`. A few common synonyms are normalized by the training script.
Multi-label rows must be reviewed and converted to a single primary label
before training; the script intentionally rejects unknown or ambiguous labels.

Include a `language` column with Whisper language codes (for example `en`,
`hi`, `bn`, `ta`, `te`, `mr`, `gu`, `pa`, `ml`, `kn`, or `ur`).
If the column is absent, the checkpoint is marked English-only. Do not claim
Hindi or multilingual support unless those languages are represented in the
training data and independently evaluated.

Do not use RAVDESS or CREMA-D audio emotion labels as ground-truth text
emotion labels. Those labels describe the acted vocal emotion and do not
necessarily describe the literal meaning of the sentence.

## 2. Potential data sources

- [GoEmotions](https://huggingface.co/datasets/mrm8488/goemotions) is an
  English Reddit-comment corpus with 27 emotion labels plus neutral. It is
  multi-label and its domain is online comments, so it requires careful
  label mapping and is not a direct proxy for spoken conversation.
- [EmoInHindi](https://www.iitp.ac.in/~ai-nlp-ml/resources.html) is a Hindi
  dialogue emotion resource with multi-label emotion and intensity annotation.
  Its resource page states research-purpose use; review the terms and paper
  before redistribution or other use.
- [HinEmo-9](https://github.com/im-rakeshsahani/hinemo9) covers English,
  Devanagari Hindi, and romanized Hinglish comments. It distinguishes a
  human-annotated gold tier from a larger model-derived tier. Keep the
  human-annotated tier for held-out evaluation and review its label taxonomy
  and license before mapping it to SpeechMind's seven classes.

These datasets differ in domain, label taxonomy, and annotation process. Do not
merge their labels mechanically. Keep a held-out evaluation set separate from
training and report results by language.

## 3. Train MuRIL

The trainer uses a stratified 80/10/10 train/validation/test split. To keep
training more practical on a CPU-only computer, it updates the final two
MuRIL encoder layers and classification head.

From the SpeechMind project folder, run:

```powershell
python -m scripts.train_text_emotion --csv data/text_emotion/text_emotion.csv
```

Optional arguments include `--epochs`, `--batch-size`, `--max-length`,
`--text-column`, and `--label-column`. The default checkpoint output is
`checkpoints/text_emotion_muril`.

The training script writes `speechmind_metadata.json` and
`text_emotion_metrics.json` alongside the checkpoint. Review test macro-F1
and per-class results before using the model. The checkpoint records which
language codes were present in the CSV; inference refuses to use the MuRIL
checkpoint for a language that was not listed.

## 4. Verify

Restart Streamlit after training. For English-only testing, the English
fallback can be loaded automatically on the first English request. For Hindi
or another Indic language, the fine-tuned MuRIL checkpoint must include that
language in its metadata.

Neither model's softmax score is a calibrated probability of correctness.
Evaluate on representative, human-labeled spoken-language transcripts before
treating outputs as reliable.


## Prepared BRIGHTER + HinEmo training path

For the prepared JSONL files produced by `PREPARE_TEXT_EMOTION_DATA.bat`, use `TRAIN_TEXT_EMOTION.bat`. It invokes `scripts/train_text_emotion_multilabel.py`, not the legacy CSV trainer. The multi-label trainer preserves the released train/dev/test splits, masks unknown HinEmo categories from both loss and validation metrics, down-weights GPT-generated HinEmo labels, and writes the six-label checkpoint format consumed by `models/text_emotion.py`. Neutral is derived only when no non-neutral signal crosses its threshold. The held-out test metrics are written to `checkpoints/text_emotion_muril/text_emotion_metrics.json`.

The default launcher uses batch size 2, max length 96, and two epochs for CPU memory constraints. Training may still take several hours on a CPU. Do not enable the checkpoint in product claims until held-out per-label results have been reviewed.


### CPU baseline configuration

The bounded CPU run uses all 2,556 BRIGHTER-Hindi training examples plus a deterministic 1,200-row HinEmo subset; one epoch; batch size 8; maximum sequence length 64; the final two MuRIL encoder layers and classification head are trainable, while lower encoder layers remain frozen. The encoder uses a 2e-5 learning rate and the classifier head uses 1e-3. Validation uses all 200 BRIGHTER dev rows plus a deterministic 500-row HinEmo subset. Test evaluation uses all 2,020 BRIGHTER test rows plus a deterministic 500-row HinEmo subset. The trainer prints progress and ETA every 25 batches. CPU training will take longer than the frozen-head baseline; review per-class prediction rates and held-out metrics before making product accuracy claims.


### Corrected CPU baseline: class balance and threshold calibration

The trainer computes bounded per-label positive-class weights from the training split.
The final two MuRIL encoder layers are fine-tuned alongside the classification head,
rather than training only a randomly initialized head. After each epoch,
it reports validation metrics at the fixed 0.50 threshold and separately sweeps
per-label thresholds from 0.05 to 0.95 using validation data only. The selected
thresholds are stored in `speechmind_metadata.json` and are then applied unchanged
to the held-out test split. Review `class_diagnostics` in the validation and test
metrics: actual positive rate, predicted positive rate, mean/max probability, and
the chosen threshold. Threshold calibration is not a substitute for good held-out
performance; do not use the model if test metrics remain poor.
