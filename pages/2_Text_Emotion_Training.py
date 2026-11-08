"""
Guided, in-app BHAAV training. No command-line steps are required.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

import streamlit as st

from scripts.train_bhaav_text_emotion import train_bhaav_model

DATA_DIR = Path("datasets/raw")
DATA_PATH = DATA_DIR / "BHAAV.zip"
DATA_URL = (
    "https://zenodo.org/records/3457467/files/"
    "Datasets-20190922T151602Z-001.zip?download=1"
)
EXPECTED_MD5 = "cdcaf1997a422f18ee12d713a17ffdea"
CHECKPOINT_DIR = Path("checkpoints/text_emotion_muril_bhaav")

st.set_page_config(page_title="Text Emotion Training", page_icon="🎙️", layout="wide")
st.title("Hindi Text Emotion Model")
st.caption("Train a MuRIL baseline on the BHAAV Hindi story-emotion corpus.")

st.warning(
    "BHAAV contains Hindi story sentences, not spontaneous speech transcripts. "
    "A model trained here is an initial text-emotion baseline, not a validated "
    "Hinglish or conversational emotion model. Use is for research with attribution; "
    "the dataset maintainers require permission for commercial use."
)

st.markdown(
    "The dataset is publicly available from Zenodo. This page downloads it directly "
    "to your computer, verifies the publisher-provided MD5 checksum, trains MuRIL, "
    "and saves the checkpoint locally. Audio files are not uploaded."
)
st.link_button("BHAAV dataset and citation", "https://zenodo.org/records/3457467")

if DATA_PATH.exists() or not st.session_state.get("bhaav_training_complete", False):
    if not DATA_PATH.exists():
        if st.button("Download BHAAV dataset", type="primary"):
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            try:
                with st.spinner("Downloading the 15.9 MB BHAAV research dataset..."):
                    request = urllib.request.Request(
                        DATA_URL,
                        headers={"User-Agent": "SpeechMind/1.0"},
                    )
                    with urllib.request.urlopen(request, timeout=120) as response:
                        payload = response.read()
                    actual_md5 = hashlib.md5(payload).hexdigest()
                    if actual_md5.lower() != EXPECTED_MD5:
                        raise ValueError(
                            "Dataset checksum mismatch. The downloaded file was not saved."
                        )
                    DATA_PATH.write_bytes(payload)
                st.success("BHAAV downloaded and checksum verified.")
                st.rerun()
            except Exception as error:
                st.error(f"Dataset download failed: {error}")
    else:
        actual_md5 = hashlib.md5(DATA_PATH.read_bytes()).hexdigest()
        if actual_md5.lower() != EXPECTED_MD5:
            st.error("The cached BHAAV archive failed checksum verification.")
            if st.button("Remove invalid cached archive"):
                DATA_PATH.unlink(missing_ok=True)
                st.rerun()
        else:
            st.success("BHAAV archive is present and checksum verified.")

if DATA_PATH.exists() and hashlib.md5(DATA_PATH.read_bytes()).hexdigest() == EXPECTED_MD5:
    if CHECKPOINT_DIR.exists() and (CHECKPOINT_DIR / "training_report.json").exists():
        st.info("A locally trained BHAAV checkpoint already exists.")
        try:
            old_report = json.loads(
                (CHECKPOINT_DIR / "training_report.json").read_text(encoding="utf-8")
            )
            st.metric(
                "Previous held-out text test macro-F1",
                f"{old_report['test_metrics']['macro_f1']:.3f}",
            )
        except Exception:
            pass

    st.subheader("Training")
    if not __import__("torch").cuda.is_available():
        st.warning(
            "CUDA is not available in this runtime. The training page will use a "
            "CPU-friendly frozen-encoder baseline; expect training to take a while. "
            "It will not fine-tune the full MuRIL encoder on CPU."
        )

    if st.button("Train / retrain MuRIL on BHAAV", type="primary"):
        status = st.status("Preparing BHAAV data and loading MuRIL...", expanded=True)
        messages = []
        try:
            def on_progress(message):
                messages.append(message)
                status.write(message)

            report = train_bhaav_model(
                DATA_PATH,
                output_dir=CHECKPOINT_DIR,
                progress_callback=on_progress,
            )
            st.session_state["bhaav_training_complete"] = True
            st.session_state["bhaav_training_report"] = report
            status.update(label="Training completed", state="complete")
            st.cache_resource.clear()
            st.success("MuRIL checkpoint saved. SpeechMind's model cache has been cleared.")
        except Exception as error:
            status.update(label="Training failed", state="error")
            st.exception(error)

report = st.session_state.get("bhaav_training_report")
if report is None and (CHECKPOINT_DIR / "training_report.json").exists():
    try:
        report = json.loads(
            (CHECKPOINT_DIR / "training_report.json").read_text(encoding="utf-8")
        )
    except Exception:
        report = None

if report:
    st.subheader("Held-out evaluation")
    left, right = st.columns(2)
    left.metric("Test accuracy", f"{report['test_metrics']['accuracy']:.3f}")
    right.metric("Test macro-F1", f"{report['test_metrics']['macro_f1']:.3f}")
    st.caption(
        "These scores are on a held-out BHAAV story-text split only. They must not be "
        "reported as speech-emotion or live-conversation accuracy."
    )
    rows = []
    for label in report["labels"]:
        item = report["test_metrics"]["classification_report"][label]
        rows.append({
            "Emotion": label.title(),
            "Precision": item["precision"],
            "Recall": item["recall"],
            "F1": item["f1-score"],
            "Support": int(item["support"]),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander("Training report and confusion matrix"):
        st.json(report)
