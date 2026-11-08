"""
SpeechMind - Streamlit demo app.

Run with:
    streamlit run app.py
"""
import tempfile
import streamlit as st

from pipeline import SpeechMindPipeline

st.set_page_config(page_title="SpeechMind", page_icon="🎙️", layout="centered")

st.title("🎙️ SpeechMind")
st.caption("Multilingual Speech, Meaning and Emotion Recognition — for Indian languages")

st.warning(
    "This system identifies emotional and stress-like vocal patterns only. "
    "It does not diagnose anxiety, depression, or any medical or psychological condition."
)


@st.cache_resource(show_spinner="Loading models (first run only)...")
def get_pipeline():
    return SpeechMindPipeline()


uploaded_file = st.file_uploader(
    "Upload a short audio clip (WAV/MP3), or record one below",
    type=["wav", "mp3", "m4a"],
)
recorded_audio = st.audio_input("Or record directly")

audio_source = uploaded_file or recorded_audio

if audio_source is not None:
    st.audio(audio_source)

    if st.button("Analyze", type="primary"):
        with st.spinner("Running pipeline..."):
            suffix = ".wav"

            if uploaded_file is not None:
                suffix = "." + uploaded_file.name.split(".")[-1].lower()

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_source.read())
                tmp_path = tmp.name

            pipeline = get_pipeline()
            result = pipeline.run(tmp_path)

        if result["low_confidence_warning"]:
            st.info(
                "⚠️ Speech recognition confidence was low for this clip — the fused "
                "result leaned more heavily on the voice-emotion signal to compensate."
            )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Language", result["language"].upper())
            st.metric("Text Emotion", result["text_emotion"].title())
            st.metric("Valence", result["valence"])
        with col2:
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            st.metric("Voice Emotion", result["voice_emotion"].title())
            st.metric("Arousal", result["arousal"])

        st.subheader("Transcript (original language)")
        st.write(result["transcript"])

        st.subheader("Meaning (English)")
        st.write(result["meaning_en"])

        st.subheader("Overall Interpretation")
        st.success(result["overall_interpretation"].title())

        with st.expander("Debug details"):
            st.json(result)
else:
    st.info("Upload or record an audio clip to get started.")