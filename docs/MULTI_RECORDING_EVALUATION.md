# Multi-recording production evaluation

SpeechMind must be evaluated on multiple speakers and recordings, not repeatedly tuned against one sample. This runner exercises the same `SpeechMindPipeline` used by the app and produces one versioned report with ASR, optional translation, optional emotion, speaker, and latency metrics.

## Quick start: HiACC test split

1. Ensure the HiACC manifest and audio have been prepared by running `PREPARE_HIACC_DATASET.bat`.
2. Run `EVALUATE_SPEECHMIND_HIACC.bat`, or run the equivalent command:

```powershell
.venv\Scripts\python.exe scripts\evaluate_speechmind_manifest.py --input "data\hiacc\manifest.csv" --audio-root "data\hiacc" --split test --limit 40 --seed 42 --max-per-speaker 1
```

The script selects up to 40 clips, approximately balanced by dataset and age/language stratum, with no more than one clip per speaker. Selection is deterministic for a fixed manifest and seed. The production pipeline and its isolated local workers are used; no hosted inference or paid service is called.

Outputs are written to a timestamped directory under `data/benchmarks/full_pipeline/`:
- `summary.json`: corpus-level metrics, subgroup metrics, speaker-overlap report, branch coverage and median timings.
- `predictions.csv`: one row per selected recording, including reference/hypothesis text, WER/CER, model route, English meaning, emotion outputs, uncertainty status and stage timings.

Use `--limit 0` to evaluate every eligible row. Use `--output-dir` to select an explicit output directory.

## Manifest schema

Required columns:

| Column | Meaning |
|---|---|
| `audio_path` | Audio file path, absolute or relative to `--audio-root` |
| `transcript` | Human reference transcript in the original spoken script(s) |
| `speaker_id` | Stable speaker ID, not a clip ID |
| `split` | Dataset-provided split, such as train, validation or test |

Optional columns:

| Column | Meaning |
|---|---|
| `dataset` | Corpus name; defaults to `unspecified` |
| `age_group` | Optional subgroup such as adult/children |
| `language` | Optional subgroup such as hi, en or hinglish |
| `reference_meaning_en` | Human-reviewed English meaning for translation scoring |
| `emotion_label` | Human gold emotion label for voice/text branch evaluation |

A blank `reference_meaning_en` or `emotion_label` means that task is **not scored** for that row. Do not fill these columns with model-generated text or model-predicted emotion and then treat them as ground truth. A blank reference is not an invitation to infer one.

A starter file is available at `docs/evaluation_manifest_template.csv`. Keep audio and full datasets out of Git.

## Speaker integrity

The summary compares speakers in the selected split with speakers found in train/validation/dev rows of the same manifest. It reports overlap and unseen-speaker counts rather than hiding leakage.

For a strict manifest-level holdout, add `--strict-speaker-holdout`. This excludes evaluation speakers that occur in train/validation/dev. It fails explicitly if the manifest cannot support that check or if no unseen speakers remain; it never falls back silently.

Important: manifest-level speaker separation does not prove that a pretrained/fine-tuned model never encountered those speakers in another source or training corpus. HiACC's accompanying dataset/model documentation discloses speaker overlap across released splits. A result on that test split must therefore be described with the actual overlap report and must not be marketed as a fully unseen-speaker result unless the model's own training speakers are known.

## How to read the scorecard

- **ASR:** micro WER/CER are computed over reference token/character totals. Scores are also broken down by dataset and age/language stratum. Multi-reference rows use the existing alternative-aware metric helper.
- **English meaning:** WER/CER are reported only for rows with human-reviewed `reference_meaning_en`. HiACC does not provide those English meaning references, so the standard HiACC run will report translation as not scored. It still records whether translation ran and its timing per clip.
- **Emotion:** voice and text outputs are evaluated separately only when `emotion_label` is supplied. The report separates coverage, abstentions, accuracy on covered predictions and accuracy across all labelled rows. An `uncertain` output is an abstention, not a correct guess.
- **Latency:** model initialization is separated from per-recording stage timings. The summary reports median and mean stage time across successful runs; compare warm runs on the same machine.
- **Failures:** a pipeline failure is retained in the predictions file and an empty ASR hypothesis is scored against the reference, rather than silently dropping a difficult clip.

The runner's successful exit means the evaluation completed. It does **not** mean the quality thresholds have passed. Record the output summary and the manifest revision alongside any claim about accuracy.

## Current evidence and remaining work

The project has a working upload/record → transcription → English meaning → text/voice emotion → uncertainty-aware interpretation path. A reference Hinglish clip has passed exact transcript comparison and its English meaning is correct. The latest warm app run took about 15.7 seconds end-to-end on the current CPU setup; ASR was still the largest stage. That is a single clip and a warm run, not a general latency guarantee.

A broad quality claim still requires:
1. a representative, speaker-aware test sample for Hindi, English and Hinglish;
2. a human-reviewed English meaning reference set;
3. a permitted human-labelled natural Indian/Hinglish speech-emotion test set;
4. per-language and per-speaker metrics with uncertainty/coverage reported;
5. a repeatable latency profile over short and long recordings.

Do not fine-tune voice emotion or change ASR routing until those evaluation results identify a specific failure. HiACC supports ASR/code-switch assessment, not emotion evaluation. BRIGHTER/HinEmo support text-emotion work with different label provenance, not vocal-emotion ground truth. RAVDESS/CREMA-D are acted English voice corpora and do not establish natural Hinglish performance.
