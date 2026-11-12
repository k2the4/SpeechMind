"""
SpeechMind — automatic multilingual speech understanding.
Run with: streamlit run app.py
"""
import os
import tempfile
import time

import streamlit as st
from pipeline import SpeechMindPipeline
from models.deployment_checkpoints import ensure_deployment_checkpoints

st.set_page_config(page_title="SpeechMind", page_icon="🎙️", layout="centered")
st.title("SpeechMind")
st.caption(
    "Speak naturally in Hindi, English, Hinglish and supported Indian languages. SpeechMind explains both what was said and the emotion conveyed through words and voice."
)
st.warning(
    "SpeechMind describes linguistic and vocal patterns. It does not diagnose "
    "anxiety, depression, or any medical or psychological condition."
)


@st.cache_resource(show_spinner="Loading core models (first run only)...")
def get_pipeline():
    ensure_deployment_checkpoints()
    return SpeechMindPipeline()


st.caption(
    "Record directly or upload a clip. SpeechMind automatically identifies the spoken language, transcribes the original speech, analyses text and vocal emotion in parallel, and presents the meaning in English."
)
st.caption(
    "Language support is component-specific. An emotion branch is only run when its installed checkpoint declares support for the detected language; unsupported analysis is marked unavailable rather than guessed."
)

uploaded_file = st.file_uploader(
    "Upload your speech (WAV/MP3/M4A), or record directly",
    type=["wav", "mp3", "m4a"],
)
recorded_audio = st.audio_input("Or record directly")
audio_source = uploaded_file or recorded_audio


def get_sarvam_api_key():
    key = os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_API_SUBSCRIPTION_KEY")
    if key:
        return key
    try:
        return (
            st.secrets.get("SARVAM_API_KEY")
            or st.secrets.get("SARVAM_API_SUBSCRIPTION_KEY")
        )
    except Exception:
        return None


selected_asr_provider = "local"
sarvam_api_key = get_sarvam_api_key()
sarvam_consent = False
if audio_source is not None and sarvam_api_key:
    provider_choice = st.radio(
        "Speech recognition engine",
        [
            "Local models (audio stays on this machine)",
            "Hosted Saaras v4 (external processing)",
        ],
        index=0,
        help=(
            "Local uses SpeechMind's Whisper/Srota/IndicConformer routes. "
            "Hosted uses Sarvam Saaras v4 in code-mixed mode."
        ),
    )
    if provider_choice.startswith("Hosted"):
        selected_asr_provider = "sarvam"
        st.warning(
            "Hosted transcription sends this recording to Sarvam AI. "
            "Use it only if you are comfortable with external audio processing."
        )
        sarvam_consent = st.checkbox(
            "I agree to send this recording to Sarvam AI for transcription."
        )

if audio_source is not None:
    st.audio(audio_source)
    if st.button(
        "Analyze",
        type="primary",
        disabled=(selected_asr_provider == "sarvam" and not sarvam_consent),
    ):
        tmp_path = None
        try:
            with st.spinner("Loading models if needed and analyzing audio..."):
                suffix = ".wav"
                if uploaded_file is not None:
                    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(audio_source.getvalue())
                    tmp_path = tmp.name

                init_start = time.perf_counter()
                pipeline = get_pipeline()
                model_init_time = round(time.perf_counter() - init_start, 3)
                if sarvam_api_key and selected_asr_provider == "sarvam":
                    # Keep the secret server-side; never expose it in the UI.
                    os.environ["SARVAM_API_KEY"] = sarvam_api_key
                result = pipeline.run(
                    tmp_path,
                    asr_provider=selected_asr_provider,
                )
                result["model_init_time_sec"] = model_init_time
        except Exception as error:
            st.error(f"Analysis failed: {error}")
            st.exception(error)
            result = None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        if result is not None:
            st.subheader("Transcript (original speech)")
            st.write(result.get("transcript") or "No transcript was produced.")
            if result.get("asr_error"):
                st.warning(
                    "Speech recognition is unavailable for this clip. "
                    + str(result.get("asr_error"))
                )

            with st.expander("Detected language spans (beta)"):
                spans = result.get("language_spans") or []
                st.caption(
                    "These are transcript character offsets, not audio timestamps. "
                    "Romanized-Hindi and English labels use conservative word-list "
                    "evidence; ambiguous Latin words remain unclassified."
                )
                if spans:
                    st.table([
                        {
                            "Text": span.get("text", ""),
                            "Language": span.get("language", "und"),
                            "Script": span.get("script", "Other"),
                            "Character range": (
                                f"{span.get('char_start', 0)}–{span.get('char_end', 0)}"
                            ),
                            "Evidence": span.get("method", ""),
                        }
                        for span in spans
                    ])
                else:
                    st.caption("No language-bearing transcript spans were detected.")

            st.subheader("Meaning (English)")
            st.write(result.get("meaning_en") or "English meaning is unavailable.")
            if result.get("meaning_en_note"):
                st.caption("Translation note: " + str(result["meaning_en_note"]))
            if not result.get("meaning_en_available", False):
                st.warning(
                    "An English translation could not be produced. The text above "
                    "is the original transcript, not an English translation. "
                    + str(result.get("translator_error") or "")
                )

            st.caption(
                "Detected primary language: "
                + str(result.get("language", "unknown")).upper()
                + ". Language labels below are conservative transcript-level estimates."
            )
            if result.get("code_switch_detected"):
                st.caption(
                    "Automatic Hindi-English code-switch route: "
                    + str(result.get("asr_model", "Srota"))
                )

            quality = result.get("asr_quality_score")
            quality_status = result.get("asr_quality_status")
            if quality_status == "unavailable":
                st.warning(
                    result.get("asr_quality_note")
                    or "Transcript quality could not be scored by this recognizer. "
                       "Check the original transcript before relying on its meaning."
                )
            elif quality is not None:
                st.caption(
                    f"Transcript quality estimate: {quality * 100:.1f}% "
                    "(uncalibrated model heuristic, not a probability of correctness)."
                )

            if result.get("low_confidence_warning"):
                reasons = result.get("uncertainty_reasons") or []
                st.info(
                    "Interpretation needs caution. "
                    + ("; ".join(reasons) if reasons else
                       "One or more model signals are uncertain or unavailable.")
                )

            st.divider()
            st.subheader("Emotion conveyed through words")
            st.metric(
                "Semantic emotion",
                str(result.get("text_emotion", "unavailable")).title()
            )
            secondary_text_emotions = result.get("text_emotion_secondary") or []
            if secondary_text_emotions:
                st.caption(
                    "Additional text signals: "
                    + ", ".join(str(label).title() for label in secondary_text_emotions)
                )
            if not result.get("text_emotion_available", False):
                st.warning(
                    "Text-emotion analysis is unavailable. "
                    + str(result.get("text_emotion_reason") or "")
                )
            st.subheader("Emotion conveyed through voice")
            st.metric(
                "Vocal emotion",
                str(result.get("voice_emotion", "unavailable")).title()
            )
            st.caption(
                "Arousal: not assessed. SpeechMind does not yet have a separately "
                "validated acoustic arousal estimator."
            )
            if not result.get("voice_branch_available", False):
                st.warning(
                    "Voice-emotion analysis is unavailable. "
                    + str(result.get("voice_emotion_reason") or "")
                )
            else:
                st.caption(
                    "Experimental baseline: the current voice model has not been "
                    "validated on spontaneous Indian/Hinglish speech."
                )
            st.divider()
            st.subheader("Overall emotional context")
            st.write(
                result.get(
                    "overall_interpretation",
                    "Combined interpretation unavailable."
                )
            )

            interpretation_status = result.get("interpretation_status")
            branch_comparison = str(result.get("branch_comparison") or "")
            if interpretation_status == "insufficient_evidence":
                st.warning(
                    "There is not enough reliable evidence to assign an overall emotion."
                )
                if branch_comparison:
                    st.caption(branch_comparison)
            elif interpretation_status == "branch_disagreement":
                st.warning(
                    "The text and voice branches produced different reliable signals. "
                    "No combined emotion or valence is assigned."
                )
                if branch_comparison:
                    st.caption(branch_comparison)
            elif interpretation_status == "single_branch":
                st.info(branch_comparison or "Only one reliable analysis branch contributed.")
                st.caption(
                    "This is a single-branch interpretation. SpeechMind does not show "
                    "a combined confidence score when both modalities have not contributed reliably."
                )
                if result.get("valence") not in (None, "Unknown"):
                    st.metric("Valence from available branch", str(result["valence"]))
            elif interpretation_status == "branch_agreement":
                st.info(branch_comparison or "Text and voice signals agree.")
                st.caption(
                    "The displayed model score is uncalibrated; it is not a probability "
                    "that the emotional interpretation is correct."
                )
                confidence = result.get("confidence")
                if confidence is not None:
                    st.metric(
                        "Combined model score (uncalibrated)",
                        f"{float(confidence) * 100:.1f}%",
                    )
                st.metric("Valence", str(result.get("valence", "unknown")))
            else:
                if branch_comparison:
                    st.caption(branch_comparison)
                st.caption(
                    "Emotion predictions are experimental and require validation "
                    "on representative held-out Indian speech."
                )

            with st.expander("Performance timings and diagnostics"):
                st.write(
                    f"Model initialization on this click: "
                    f"{result.get('model_init_time_sec', 0.0)} s"
                )
                st.json({
                    "stage_timings_sec": result.get("timings_sec", {}),
                    "asr_model": result.get("asr_model"),
                    "asr_provider": result.get("asr_provider"),
                    "asr_error": result.get("asr_error"),
                    "code_switch_detected": result.get("code_switch_detected"),
                    "language_spans": result.get("language_spans"),
                    "language_evidence": result.get("language_evidence"),
                    "script_evidence": result.get("script_evidence"),
                    "languages_present": result.get("languages_present"),
                    "scripts_present": result.get("scripts_present"),
                    "mixed_script_detected": result.get("mixed_script_detected"),
                    "romanized_hindi_detected": result.get("romanized_hindi_detected"),
                    "language_span_unit": result.get("language_span_unit"),
                    "audio_timestamps_available": result.get("audio_timestamps_available"),
                    "asr_candidate_model": result.get("asr_candidate_model"),
                    "asr_candidate_inference_seconds": result.get(
                        "asr_candidate_inference_seconds"
                    ),
                    "asr_inference_device": result.get("asr_inference_device"),
                    "language_source": result.get("language_source"),
                    "language_detection_score": result.get("language_detection_score"),
                    "asr_quality_score": result.get("asr_quality_score"),
                    "asr_quality_status": result.get("asr_quality_status"),
                    "asr_quality_note": result.get("asr_quality_note"),
                    "asr_quality_method": result.get("asr_quality_method"),
                    "asr_avg_logprob": result.get("asr_avg_logprob"),
                    "asr_avg_no_speech_prob": result.get("asr_avg_no_speech_prob"),
                    "asr_avg_compression_ratio": result.get("asr_avg_compression_ratio"),
                    "text_emotion_model_source": result.get("text_emotion_model_source"),
                    "text_emotion_available": result.get("text_emotion_available"),
                    "text_emotion_reason": result.get("text_emotion_reason"),
                    "text_emotion_secondary": result.get("text_emotion_secondary"),
                    "text_emotion_multi_label_scores": result.get(
                        "text_emotion_multi_label_scores"
                    ),
                    "text_emotion_task_type": result.get("text_emotion_task_type"),
                    "voice_model_source": result.get("voice_model_source"),
                    "voice_emotion_available": result.get("voice_emotion_available"),
                    "voice_emotion_reason": result.get("voice_emotion_reason"),
                    "voice_inference_device": result.get("voice_inference_device"),
                    "voice_inference_windows": result.get("voice_inference_windows"),
                    "voice_validation_macro_f1": result.get("voice_validation_macro_f1"),
                    "fusion_weights": result.get("fusion_weights"),
                    "branch_agreement": result.get("branch_agreement"),
                    "branch_disagreement": result.get("branch_disagreement"),
                    "branch_comparison": result.get("branch_comparison"),
                    "interpretation_status": result.get("interpretation_status"),
                    "uncertainty_reasons": result.get("uncertainty_reasons"),
                    "confidence_margin": result.get("confidence_margin"),
                    "arousal_available": result.get("arousal_available"),
                    "meaning_en_available": result.get("meaning_en_available"),
                    "meaning_en_source": result.get("meaning_en_source"),
                    "meaning_en_source_language": result.get("meaning_en_source_language"),
                    "meaning_en_method": result.get("meaning_en_method"),
                    "meaning_en_note": result.get("meaning_en_note"),
                    "translator_provider": result.get("translator_provider"),
                    "translator_loaded": result.get("translator_loaded"),
                    "translator_inference_seconds": result.get(
                        "translator_inference_seconds"
                    ),
                    "translator_error": result.get("translator_error"),
                    "full_result": result,
                })
else:
    st.info("Upload or record an audio clip to get started.")
