"""
SpeechMind - Streamlit demo.
Run with: streamlit run app.py

Language identification is automatic. Users never have to choose Hindi,
English, or Hinglish before speaking.
"""
import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from pipeline import SpeechMindPipeline

load_dotenv()
st.set_page_config(page_title="SpeechMind", page_icon="🎙️", layout="centered")
st.title("SpeechMind")
st.caption("Multilingual Speech Understanding for Indian languages and natural code-switching")
st.warning(
    "This system describes linguistic meaning and vocal expression. "
    "It does not diagnose anxiety, depression, stress disorders, or any medical "
    "or psychological condition."
)
st.caption(
    "Speak naturally. SpeechMind identifies the language automatically; "
    "you do not need to select Hindi, English, or Hinglish."
)


@st.cache_resource(show_spinner="Loading core models (first run only)...")
def get_pipeline():
    return SpeechMindPipeline()


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
            with st.spinner("Analyzing the recording..."):
                suffix = ".wav"
                if uploaded_file is not None:
                    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(audio_source.getvalue())
                    tmp_path = tmp.name

                init_start = time.perf_counter()
                pipeline = get_pipeline()
                model_init_time = round(time.perf_counter() - init_start, 3)
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
            if result["low_confidence_warning"]:
                st.info(
                    "Speech-recognition quality is low or unavailable. "
                    "The combined result gives less weight to the text branch."
                )
            if not result.get("text_emotion_available", False):
                st.warning(
                    "Text-emotion analysis is unavailable for this transcript. "
                    + str(result.get("text_emotion_reason") or "")
                )
            if not result.get("translation_available", False):
                st.warning(
                    "English meaning is unavailable. "
                    + str(result.get("translation_reason") or
                          "Configure a code-mixed translation provider for this recording.")
                )

            left, right = st.columns(2)
            with left:
                st.metric("Detected language", str(result.get("language", "unknown")).upper())
                st.metric("Text emotion", result["text_emotion"].title())
                st.metric("Valence", result["valence"])
            with right:
                st.metric("Combined score (uncalibrated)", f"{result['confidence'] * 100:.1f}%")
                st.metric("Voice emotion", result["voice_emotion"].title())
                st.metric("Vocal arousal (experimental)", result["arousal"])

            if result.get("signals_agree") is False:
                st.info(
                    "Text and voice signals differ. SpeechMind preserves both findings "
                    "instead of treating the disagreement as an error."
                )

            quality = result.get("asr_quality_score")
            if quality is not None:
                st.caption(
                    f"ASR token-quality heuristic: {quality * 100:.1f}% "
                    "(not a probability that the transcript is correct)."
                )
            else:
                st.caption("ASR quality score is unavailable for the selected decoder.")

            st.caption(
                "The combined score is not a calibrated probability of correctness. "
                "Treat all predictions as experimental until validated on representative data."
            )

            st.subheader("Original transcript")
            st.write(result["transcript"] or "No transcript was produced.")
            st.subheader("Meaning in English")
            st.write(result["meaning_en"] or "No English meaning was produced.")

            st.subheader("Text and voice findings")
            text_col, voice_col = st.columns(2)
            with text_col:
                st.markdown("**Text analysis**")
                st.write(result["text_emotion"].title())
                st.caption(result.get("text_emotion_model_source") or
                           result.get("text_emotion_reason") or "Text model information unavailable.")
            with voice_col:
                st.markdown("**Voice analysis**")
                st.write(result["voice_emotion"].title())
                st.caption(
                    f"Model: {result.get('voice_model_source') or 'unknown'}"
                )

            st.subheader("Combined interpretation")
            st.write(result["overall_interpretation"])

            with st.expander("Performance timings and diagnostics"):
                st.write(f"Model initialization on this click: {result['model_init_time_sec']} s")
                st.json({
                    "stage_timings_sec": result.get("timings_sec", {}),
                    "asr_model": result.get("asr_model"),
                    "asr_backend": result.get("asr_backend"),
                    "language_source": result.get("language_source"),
                    "languages_detected": result.get("languages_detected"),
                    "asr_quality_score": result.get("asr_quality_score"),
                    "asr_quality_method": result.get("asr_quality_method"),
                    "asr_avg_logprob": result.get("asr_avg_logprob"),
                    "asr_avg_no_speech_prob": result.get("asr_avg_no_speech_prob"),
                    "asr_avg_compression_ratio": result.get("asr_avg_compression_ratio"),
                    "asr_chunk_count": result.get("asr_chunk_count"),
                    "translation_backend": result.get("translation_backend"),
                    "translation_available": result.get("translation_available"),
                    "translation_reason": result.get("translation_reason"),
                    "text_emotion_model_source": result.get("text_emotion_model_source"),
                    "text_emotion_available": result.get("text_emotion_available"),
                    "text_emotion_reason": result.get("text_emotion_reason"),
                    "voice_model_source": result.get("voice_model_source"),
                    "voice_inference_device": result.get("voice_inference_device"),
                    "voice_inference_windows": result.get("voice_inference_windows"),
                    "voice_validation_macro_f1": result.get("voice_validation_macro_f1"),
                    "signals_agree": result.get("signals_agree"),
                    "signal_agreement": result.get("signal_agreement"),
                    "fusion_weights": result.get("fusion_weights"),
                })
                st.json(result)
else:
    st.info("Upload or record audio to get started.")
