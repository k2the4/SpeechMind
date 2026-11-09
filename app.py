"""
SpeechMind - Streamlit demo.
Run with: streamlit run app.py
"""
import os
import tempfile
import time

import streamlit as st
from pipeline import SpeechMindPipeline

st.set_page_config(page_title="SpeechMind", page_icon="🎙️", layout="centered")
st.title("SpeechMind")
st.caption("Multilingual Speech, Meaning and Emotion Recognition — for Indian languages")
st.warning(
    "This system identifies emotional and stress-like vocal patterns only. "
    "It does not diagnose anxiety, depression, or any medical or psychological condition."
)


@st.cache_resource(show_spinner="Loading core models (first run only)...")
def get_pipeline():
    return SpeechMindPipeline()


LANGUAGE_OPTIONS = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Odia": "or",
    "Punjabi": "pa",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
}

selected_language = st.selectbox(
    "Speech language",
    options=list(LANGUAGE_OPTIONS),
    index=0,
    help="For short clips or unclear accents, select the language being spoken instead of relying on automatic detection.",
)
language_hint = LANGUAGE_OPTIONS[selected_language]

uploaded_file = st.file_uploader(
    "Upload a short audio clip (WAV/MP3/M4A), or record one below",
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
                result = pipeline.run(tmp_path, language_hint=language_hint)
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
            if result["low_confidence_warning"]:
                st.info(
                    "Speech-recognition quality is low or unavailable. The final result "
                    "relied more heavily on the voice signal."
                )
            if not result.get("text_emotion_available", False):
                st.warning(
                    "Text-emotion analysis is unavailable for this language. "
                    + str(result.get("text_emotion_reason") or "")
                )

            left, right = st.columns(2)
            with left:
                st.metric("Language", result["language"].upper())
                st.metric("Text Emotion", result["text_emotion"].title())
                st.metric("Valence", result["valence"])
            with right:
                st.metric("Fusion score (uncalibrated)", f"{result['confidence'] * 100:.1f}%")
                st.metric("Voice Emotion", result["voice_emotion"].title())
                st.metric("Arousal", result["arousal"])

            quality = result.get("asr_quality_score")
            if quality is not None:
                st.caption(
                    f"ASR token-quality heuristic: {quality * 100:.1f}% "
                    "(not a probability that the transcript is correct)."
                )
            else:
                st.caption("ASR quality score is unavailable for this decoder.")

            st.caption(
                "The fusion score is not a calibrated probability of correctness. "
                "Predictions are experimental until validated on held-out, real-world data."
            )
            st.subheader("Transcript (original language)")
            st.write(result["transcript"] or "No transcript was produced.")
            st.subheader("Meaning (English)")
            st.write(result["meaning_en"] or "No translation was produced.")
            st.subheader("Overall Interpretation")
            st.write(result["overall_interpretation"])

            with st.expander("Performance timings and diagnostics"):
                st.write(f"Model initialization on this click: {result['model_init_time_sec']} s")
                st.json({
                    "stage_timings_sec": result.get("timings_sec", {}),
                    "asr_model": result.get("asr_model"),
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
                    "translator_loaded": result.get("translator_loaded"),
                    "translator_error": result.get("translator_error"),
                })
                st.json(result)
else:
    st.info("Upload or record an audio clip to get started.")
