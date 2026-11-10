# SpeechMind V2 — Architecture and Engineering Plan

This architecture implements the canonical product contract in `SPEECHMIND_PRODUCT_SPEC.md`. Model names are candidates until benchmarked on representative data; they are not permanent product requirements.

## 1. End-to-end data flow

```text
Audio input
   |
   v
Audio decode / quality checks / conservative VAD
   |
   +-------------------------------> Preserve waveform ------------------+
   |                                                                      |
   v                                                                      v
Automatic code-switch ASR                                      Vocal analysis
   |                                                          (WavLM / emotion2vec)
   v                                                                      |
Original transcript                                                       |
   |                                                                      |
   +----------------------+                                               |
   |                      |                                               |
   v                      v                                               |
Text emotion        Code-mixed-to-English translation                    |
   |                      |                                               |
   +----------------------+----------------------+------------------------+
                                                  |
                                                  v
                                 Reliability-aware multimodal fusion
                                                  |
                                                  v
                      Original transcript + English meaning + separate
                      text/voice findings + combined interpretation
```

## 2. Audio input and preparation

Input is decoded to mono 16 kHz float audio for model compatibility. Preserve the complete decoded waveform for voice analysis. Use conservative VAD for long-file chunking, keeping boundary padding and joining metadata. Do not hard-trim quiet pauses from the waveform used by the voice model.

Audio quality diagnostics may include clipping, duration, silence ratio, and sample-rate conversion. They are diagnostics, not automatic grounds for discarding a person's recording.

## 3. ASR layer: provider interface, not one hard-coded model

The ASR component must expose one stable interface so providers can be benchmarked without rewriting the pipeline.

Initial candidates:
- **Indic-Transcribe Flex / Core:** designed for Indian speech, accents, and code-mixing. Flex supports mixed/native/Roman script renderings. Downloaded checkpoints are large; hosted inference is a practical evaluation path on the current CPU-only / GT 730 PC.
- **Srota (Qwen3-ASR Hinglish fine-tune):** a specialist candidate for natural Hindi-English output in mixed Devanagari + Latin script. Its model card reports in-domain results on HiACC and OpenSLR-derived data; its stated limitations and split caveats must be respected.
- **Whisper Small:** existing local fallback and baseline. It must run in automatic multilingual transcription mode rather than first forcing the entire recording into a single detected language.
- **Bodhan Indic-Transcribe API:** optional hosted provider for code-switch and multi-language evaluation. It is only used when explicitly configured; audio is sent to the external service in that mode.

Do not choose a production default by model-card claims alone. Benchmark on the same held-out audio and references. Report WER/CER by domain and language. A model that wins on Hinglish conversational speech may not win on monolingual Tamil or technical/tutorial speech.

### Provider/privacy configuration

- `SPEECHMIND_ASR_BACKEND=whisper` is the privacy-first local default.
- `SPEECHMIND_ASR_BACKEND=bodhan` explicitly sends audio to Bodhan's hosted transcription API.
- API keys are stored in local `.env`, never committed to Git.
- Hosted provider failure must be visible. Do not silently send audio to another provider or silently switch to a cloud provider.

The API accepts audio requests up to 30 seconds; long recordings need VAD-aware chunking and careful transcript joining. The provider's returned language/quality metadata must be used only when actually supplied.

## 4. Transcript representation

Represent ASR output internally as a structured object:
- `text`: original transcript.
- `language`: one detected language only when reliable; otherwise `mixed` or `unknown`.
- `language_segments`: optional time/token-level language labels, if the provider supplies them.
- `segments`: timestamped text segments when available.
- `backend`, `quality_score`, `quality_method`, and raw diagnostics.

Do not fabricate token-level labels from a single file-level language guess. Preserve the original source transcript exactly; display normalization belongs in a separate view.

## 5. English meaning / translation

Preferred candidate: **Bodhan Indic-Translate** because its model/API documentation explicitly covers romanized Indic input and code-mixed text and does not require the caller to provide a single source language. The source must be passed as-is, with target language English.

Local fallback: existing distilled IndicTrans2 for supported, clearly monolingual Indic-script input. It should not be treated as a reliable code-mixed translator simply because it can translate Hindi.

If code-mixed translation is unavailable, report that clearly. Do not present the original Hinglish transcript as if it were an English translation.

Transliteration is a separate task from translation and must never be substituted for English meaning conversion.

## 6. Text understanding

The text model must receive the original transcript. If a separate English meaning is available, it may be supplied as a second feature/view, but never instead of the source transcript.

Candidates:
- MuRIL fine-tuned on Hindi/romanized Hindi/English code-mixed text.
- XLM-R as a controlled comparison.

The HinEmo-9 dataset is a useful candidate for code-mixed text-emotion research, but its gold vs model-derived tiers and nine-label taxonomy must be respected. Human-gold labels are the evaluation reference; model-derived labels must be marked as noisy training data. Do not map culturally distinct labels such as devotion/nostalgia into joy/sadness without an explicit, reviewed annotation policy.

Prefer multi-label emotion evidence where an utterance can convey more than one emotion. Keep neutral/unknown handling explicit.

## 7. Vocal understanding

Keep the existing WavLM + SVM checkpoint as the initial baseline. Benchmark emotion2vec+ base as an alternative representation/model. The emotion2vec model card exposes nine outputs, including other and unknown; those labels must not be silently forced into SpeechMind's seven-label taxonomy.

Evaluate on speaker-independent tests and report by corpus. RAVDESS/CREMA-D scores are acted-English baselines, not evidence of natural Indian speech performance.

The current arousal shortcut derived from the predicted emotion distribution is provisional. A proper arousal indicator needs independently validated acoustic/prosodic features or a separately trained model.

## 8. Fusion

Keep separate:
- text emotion distribution,
- voice emotion distribution,
- ASR quality,
- translation availability/quality,
- agreement/disagreement status,
- combined interpretation.

Current 60/40-style weighting is a provisional baseline only. Future weights should be learned/calibrated using representative paired audio, transcripts, and human labels, not selected by intuition. Fusion must support:
- agreement;
- positive text / negative or subdued voice;
- negative text / calm voice;
- one unavailable branch;
- low-quality transcript;
- low-confidence voice;
- inconclusive result.

A final score is not a probability of correctness unless calibrated and evaluated as such.

## 9. Deployment tiers

**Development on the current second PC:** CPU-only PyTorch, 16 GB RAM, GT 730. Keep Whisper and the existing WavLM model as local baselines. Do not assume a 1.2B FP16 ASR checkpoint or 7.94B translation model will run well locally.

**Code-switch evaluation:** Use optional hosted Bodhan ASR/translation providers only after the user configures keys and accepts sending recordings externally; cost and privacy implications must be visible.

**Research/training:** Use a separate Colab/Kaggle or GPU environment for large ASR fine-tuning. Do not contaminate the working Windows environment with experimental dependency versions. Keep experimental model installs in a separate environment.

## 10. Release gates

No component becomes the default merely because it loads. The release gate is:
1. same evaluation manifest for candidate comparison;
2. per-dataset/per-language WER/CER or emotion metrics;
3. speaker separation where relevant;
4. explicit model/data/license provenance;
5. reproducible local setup or clearly documented API key setup;
6. successful unit/CI tests;
7. a small human-reviewed real-world Hinglish evaluation set.
