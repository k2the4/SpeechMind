"""
SpeechMind — automatic multilingual speech understanding.
Run with: streamlit run app.py
"""
import os
import tempfile
import time

import streamlit as st
from pipeline import SpeechMindPipeline

st.set_page_config(page_title="SpeechMind", page_icon="🎙️", layout="centered")
st.title("SpeechMind")
st.caption(
    "Understand natural Indian speech — English, Hindi, Hinglish and selected Indian languages."
)
st.warning(
    "SpeechMind describes linguistic and vocal patterns. It does not diagnose "
    "anxiety, depression, or any medical or psychological condition."
)


@st.cache_resource(show_spinner="Loading core models (first run only)...")
def get_pipeline():
    return SpeechMindPipeline()


st.caption(
    "No language selection is needed. Speak naturally; SpeechMind attempts to "
    "recognize the languages used in your recording."
)

uploaded_file = st.file_uploader(
    "Upload an audio clip (WAV/MP3/M4A), or record one below",
    type=["wav", "mp3", "m4a"],
)
recorded_audio = st.audio_input("Or record directly")
audio_source = uploaded_file or recorded_audio

if audio_source is not None:
    st.audio(audio_source)
    if st.button("Analyze", type="primary"):
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
                # Always use automatic language identification. The UI must not
                # force users to label naturally code-mixed speech.
                result = pipeline.run(tmp_path)
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

            st.subheader("Meaning (English)")
            st.write(result.get("meaning_en") or "English meaning is unavailable.")
            if not result.get("meaning_en_available", False):
                st.warning(
                    "An English translation could not be produced. The text above "
                    "is the original transcript, not an English translation. "
                    + str(result.get("translator_error") or "")
                )

            st.caption(
                "Detected primary language: "
                + str(result.get("language", "unknown")).upper()
                + ". Mixed-language span detection is still being developed."
            )
            if result.get("code_switch_detected"):
                st.caption(
                    "Automatic Hindi-English code-switch route: "
                    + str(result.get("asr_model", "Srota"))
                )

            quality = result.get("asr_quality_score")
            if quality is not None:
                st.caption(
                    f"ASR token-quality indicator: {quality * 100:.1f}% "
                    "(uncalibrated; not a probability that the transcript is correct)."
                )
            else:
                st.caption("ASR quality information is unavailable for this decoder.")

            if result.get("low_confidence_warning"):
                st.info(
                    "Transcription quality is uncertain. SpeechMind has reduced "
                    "the influence of the text branch where possible."
                )

            st.divider()
            st.subheader("Text analysis")
            st.metric("Semantic emotion", str(result.get("text_emotion", "unavailable")).title())
            if not result.get("text_emotion_available", False):
                st.warning(
                    "Text-emotion analysis is unavailable. "
                    + str(result.get("text_emotion_reason") or "")
                )

            st.subheader("Voice analysis")
            voice_col, arousal_col = st.columns(2)
            with voice_col:
                st.metric("Vocal emotion", str(result.get("voice_emotion", "unavailable")).title())
            with arousal_col:
                st.metric("Arousal", str(result.get("arousal", "unknown")))

            st.divider()
            st.subheader("Combined interpretation")
            st.write(result.get("overall_interpretation", "Combined interpretation unavailable."))
            st.caption(
                "The fusion score is an uncalibrated model score, not a probability "
                "of correctness. Predictions remain experimental until validated on "
                "representative held-out Indian speech."
            )
            st.metric(
                "Combined model score (uncalibrated)",
                f"{float(result.get('confidence', 0.0)) * 100:.1f}%",
            )
            st.metric("Valence", str(result.get("valence", "unknown")))

            with st.expander("Performance timings and diagnostics"):
                st.write(
                    f"Model initialization on this click: "
                    f"{result.get('model_init_time_sec', 0.0)} s"
                )
                st.json({
                    "stage_timings_sec": result.get("timings_sec", {}),
                    "asr_model": result.get("asr_model"),
                    "code_switch_detected": result.get("code_switch_detected"),
                    "asr_candidate_model": result.get("asr_candidate_model"),
                    "asr_candidate_inference_seconds": result.get(
                        "asr_candidate_inference_seconds"
                    ),
                    "language_source": result.get("language_source"),
                    "language_detection_score": result.get("language_detection_score"),
                    "asr_quality_score": result.get("asr_quality_score"),
                    "asr_quality_method": result.get("asr_quality_method"),
                    "asr_avg_logprob": result.get("asr_avg_logprob"),
                    "asr_avg_no_speech_prob": result.get("asr_avg_no_speech_prob"),
                    "asr_avg_compression_ratio": result.get("asr_avg_compression_ratio"),
                    "text_emotion_model_source": result.get("text_emotion_model_source"),
                    "text_emotion_available": result.get("text_emotion_available"),
                    "text_emotion_reason": result.get("text_emotion_reason"),
                    "voice_model_source": result.get("voice_model_source"),
                    "voice_inference_device": result.get("voice_inference_device"),
                    "voice_inference_windows": result.get("voice_inference_windows"),
                    "voice_validation_macro_f1": result.get("voice_validation_macro_f1"),
                    "fusion_weights": result.get("fusion_weights"),
                    "meaning_en_available": result.get("meaning_en_available"),
                    "meaning_en_source": result.get("meaning_en_source"),
                    "translator_loaded": result.get("translator_loaded"),
                    "translator_inference_seconds": result.get(
                        "translator_inference_seconds"
                    ),
                    "translator_error": result.get("translator_error"),
                    "full_result": result,
                })
else:
    st.info("Upload or record an audio clip to get started.")
