"""
SpeechMind - Streamlit demo.
Run with: streamlit run app.py
"""
import os
import tempfile
import time

import streamlit as st
from pipeline import SpeechMindPipeline
from models.asr import INDIC_LANGUAGES

st.set_page_config(page_title="SpeechMind", page_icon="🎙️", layout="centered")
st.title("SpeechMind")
st.caption("Multilingual Speech, Meaning and Emotion Recognition — for Indian languages")
st.warning(
    "This system identifies emotional and stress-like vocal patterns only. "
    "It does not diagnose anxiety, depression, or any medical or psychological condition."
)


@st.cache_resource(show_spinner="Preparing SpeechMind (first run only)...")
def get_pipeline():
    return SpeechMindPipeline()


language_labels = {"Auto-detect": None, "English": "en"}
language_labels.update({
    name: code for code, name in INDIC_LANGUAGES.items()
})
selected_language = st.selectbox(
    "Spoken language",
    list(language_labels.keys()),
    index=0,
    help=(
        "Auto-detect is the default. Selecting the language you are speaking "
        "skips the Whisper language-detection pass and loads only the relevant "
        "speech-recognition model."
    ),
)

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
                result = pipeline.run(
                    tmp_path,
                    language_hint=language_labels[selected_language],
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
            if result["low_confidence_warning"]:
                st.info(
                    "Speech-recognition score is low. The final result relied more heavily "
                    "on the voice signal."
                )
            if result.get("voice_emotion_language_warning", False):
                st.warning(
                    "Voice-emotion analysis is disabled for this language because "
                    "the available checkpoint is English-only. It is not applied "
                    "to Indian-language recordings."
                )
            if result.get("text_emotion_available", False):
                st.info(
                    "Text emotion uses a BHAAV-trained Hindi story-text baseline. "
                    "It has not been validated for spontaneous conversation transcripts."
                )
            if not result.get("text_emotion_available", False):
                reason = result.get("text_emotion_reason") or "No compatible trained checkpoint is available."
                st.warning(
                    f"Text-emotion analysis is unavailable: {reason} "
                    "The final result excludes the text branch."
                )

            if not result.get("emotion_analysis_available", False):
                st.warning(
                    "No emotion prediction is available for this recording yet. "
                    "Speech recognition and translation can still be used."
                )

            left, right = st.columns(2)
            with left:
                st.metric("Language", result["language"].upper())
                st.metric("Text Emotion", result["text_emotion"].title())
                st.metric("Valence", result["valence"])
            with right:
                score_label = (
                    f"{result['confidence'] * 100:.1f}%"
                    if result.get("emotion_analysis_available", False)
                    else "Unavailable"
                )
                st.metric("Model score (uncalibrated)", score_label)
                st.metric("Voice Emotion", result["voice_emotion"].title())
                st.metric("Arousal", result["arousal"])

            if result.get("emotion_analysis_available", False):
                st.caption(
                    "The model score is not a calibrated probability of correctness. "
                    "Treat predictions as experimental until validated on held-out data."
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
                    "language_route": result.get("language_route"),
                    "asr_model": result.get("asr_model"),
                    "voice_emotion_model_language_scope": result.get("voice_emotion_model_language_scope"),
                    "asr_score_uncalibrated": result.get("asr_confidence"),
                    "asr_confidence_method": result.get("asr_confidence_method"),
                    "fusion_weights": result.get("fusion_weights"),
                    "text_emotion_available": result.get("text_emotion_available"),
                    "text_model_loaded": result.get("text_model_loaded"),
                    "voice_model_loaded": result.get("voice_model_loaded"),
                    "emotion_analysis_available": result.get("emotion_analysis_available"),
                    "text_emotion_model_domain": result.get("text_emotion_model_domain"),
                    "translator_loaded": result.get("translator_loaded"),
                    "translator_error": result.get("translator_error"),
                })
                st.json(result)
else:
    st.info("Upload or record an audio clip to get started.")
