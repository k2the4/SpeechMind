import streamlit as st

from utils.dataset_audit import audit_local_datasets

st.set_page_config(page_title="SpeechMind Dataset Inventory", page_icon="🎙️", layout="wide")
st.title("Dataset Inventory")
st.caption(
    "Read-only scan of the local Test Data folder. Audio remains on this computer; "
    "nothing is uploaded or modified."
)

report = audit_local_datasets()

if report["root_count"] == 0:
    st.warning(
        "No Test Data folder was found in the known Desktop/OneDrive locations. "
        "The inventory cannot inspect files outside the locations configured in this page."
    )
else:
    st.write("Scanned location(s):")
    for root in report["roots"]:
        st.code(root)

    a, b, c, d = st.columns(4)
    a.metric("Total files", report["total_files"])
    b.metric("WAV files", report["wav_files"])
    c.metric("CSV metadata", report["csv_files"])
    d.metric("Other audio", report["other_audio_files"])

    st.subheader("RAVDESS speech dataset")
    if report["ravdess_speech_wavs"]:
        x, y = st.columns(2)
        x.metric("Recognized RAVDESS speech WAVs", report["ravdess_speech_wavs"])
        y.metric("Actors found", f"{report['ravdess_actor_count']} / 24")
        st.write("Emotion-label counts parsed from filenames:")
        st.dataframe(
            [{"Emotion": label.title(), "Files": count}
             for label, count in report["ravdess_emotion_counts"].items()],
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            "RAVDESS is useful for checking the ASR pipeline because its two spoken "
            "sentences are encoded in each filename. However, the currently configured "
            "voice-emotion checkpoint reports RAVDESS as part of its training data, so "
            "using these same files to report its test accuracy would be data leakage."
        )
    else:
        st.write("No RAVDESS-formatted speech files were recognized.")

    st.subheader("Hindi audio")
    if report["hindi_folder_found"]:
        st.metric("WAV files in a Hindi folder", report["hindi_folder_wavs"])
        st.info(
            "These files can help test Hindi transcription if transcripts are available. "
            "They are not automatically emotion-labelled examples."
        )
    else:
        st.write("No Hindi subfolder containing WAV files was found.")

    st.subheader("Data readiness")
    st.write(
        "**Voice-emotion training:** use an Indian speech-emotion corpus with reliable "
        "labels and speaker metadata. BhavVani audio is available by request for "
        "non-commercial research; IITKGP-SEHSC is another request-based option."
    )
    st.write(
        "**Text-emotion training:** BHAAV is a Hindi text corpus with five classes. "
        "It is useful as a Hindi text baseline, but its sentences come from stories, "
        "not spontaneous speech transcripts, so that domain difference must be reported."
    )

if report["warnings"]:
    st.subheader("Notes")
    for warning in report["warnings"]:
        st.warning(warning)

with st.expander("Raw inventory details"):
    st.json(report)
