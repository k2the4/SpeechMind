"""
SpeechMind — multilingual customer-support conversation intelligence.
Run with: streamlit run app.py
"""
import hashlib
import json
import os
import tempfile
import time

import streamlit as st
from models.deployment_checkpoints import ensure_deployment_checkpoints
from conversation_insights import analyze_conversation

st.set_page_config(
    page_title="SpeechMind | Call Review",
    page_icon="S",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem;}
    .sm-kicker {font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
        color:#818cf8; font-weight:700; margin-bottom:.25rem;}
    .sm-subtitle {font-size:1rem; color:var(--text-color); opacity:.72; margin-top:-.45rem;}
    .sm-card {border:1px solid rgba(128,128,128,.20); border-radius:12px;
        padding:1rem 1.1rem; background:rgba(128,128,128,.045); min-height:100px;}
    .sm-label {font-size:.78rem; color:var(--text-color); opacity:.65; margin-bottom:.35rem;}
    .sm-value {font-size:1.35rem; font-weight:650; line-height:1.25;}
    .sm-section-note {color:var(--text-color); opacity:.65; font-size:.88rem;}
    div[data-testid="stMetric"] {background:rgba(128,128,128,.045);
        border:1px solid rgba(128,128,128,.16); padding:12px 14px; border-radius:10px;}
    div[data-testid="stTabs"] button {font-weight:600;}
    div[data-testid="stAlert"] {border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="sm-kicker">CALL REVIEW WORKSPACE</div>', unsafe_allow_html=True)
st.title("SpeechMind")
st.markdown(
    '<div class="sm-subtitle">Turn multilingual support calls into clear, evidence-backed review notes.</div>',
    unsafe_allow_html=True,
)
st.write("")
with st.expander("About this analysis and its limitations"):
    st.caption(
        "SpeechMind identifies transcript-based topics and review cues to support a human reviewer. "
        "A free local Ollama model provides contextual analysis when installed; otherwise the app uses "
        "deterministic fallback rules. Evidence quotes are checked against the original transcript. "
        "It does not verify complaints, confirm that an issue is resolved, or infer a person's true "
        "emotional state. Language and model support varies by component."
    )


@st.cache_resource(show_spinner="Loading SpeechMind models (first run only)...")
def get_pipeline():
    # Keep heavyweight ML imports out of the initial web-app startup path.
    ensure_deployment_checkpoints()
    from pipeline import SpeechMindPipeline

    return SpeechMindPipeline()


st.subheader("1. Add a support call")
st.caption("Upload an audio file or record a call excerpt. Supported uploads: WAV, MP3 and M4A.")
uploaded_file = st.file_uploader(
    "Audio file", type=["wav", "mp3", "m4a"], label_visibility="collapsed"
)
recorded_audio = st.audio_input("Or record directly")
audio_source = uploaded_file or recorded_audio

if audio_source is not None:
    audio_bytes = audio_source.getvalue()
    audio_digest = hashlib.sha256(audio_bytes).hexdigest()
    if st.session_state.get("speechmind_audio_digest") != audio_digest:
        st.session_state["speechmind_audio_digest"] = audio_digest
        for key in ("speechmind_result", "speechmind_model_init_time", "speechmind_insights", "speechmind_insights_digest"):
            st.session_state.pop(key, None)
    st.audio(audio_source)
    analysis_mode = st.radio(
        "Processing mode",
        ["Full SpeechMind analysis", "Fast support case intake"],
        horizontal=True,
        help=(
            "Fast mode runs the existing ASR and support-case extraction, while skipping "
            "English translation and optional emotion inference for this run. Full mode "
            "runs all existing SpeechMind analysis branches."
        ),
    )
    fast_case_mode = analysis_mode == "Fast support case intake"
    if st.button("Analyze call", type="primary", use_container_width=False):
        tmp_path = None
        try:
            with st.spinner("Analyzing the call. The first run may take longer while models load..."):
                suffix = ".wav"
                if uploaded_file is not None:
                    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                init_start = time.perf_counter()
                pipeline = get_pipeline()
                model_init_time = round(time.perf_counter() - init_start, 3)
                # Fast support intake must use the single-pass Whisper route.
                # This avoids gated IndicConformer and the unavailable Srota worker
                # on Streamlit Cloud while preserving specialist routing in Full mode.
                os.environ["SPEECHMIND_ASR_FAST_MODE"] = "1" if fast_case_mode else "0"
                print(
                    "[Deployment] ASR route: "
                    + ("single-pass Whisper (fast intake)" if fast_case_mode else "full specialist routing")
                )
                result = pipeline.run(
                    tmp_path,
                    include_emotion=not fast_case_mode,
                    include_translation=not fast_case_mode,
                )
                # On hosted fast intake, use Gemini's audio understanding as a
                # transcript fallback when Whisper's heuristic quality is weak.
                # Keep the raw Whisper hypothesis for audit and never overwrite it.
                if fast_case_mode and float(result.get("asr_quality_score") or 0.0) < 0.60:
                    try:
                        from local_llm import transcribe_audio_with_gemini
                        whisper_transcript = str(result.get("transcript") or "")
                        gemini_transcript = transcribe_audio_with_gemini(
                            tmp_path, language_hint=result.get("language", "auto")
                        )
                        if gemini_transcript.strip():
                            result["asr_whisper_transcript"] = whisper_transcript
                            result["transcript"] = gemini_transcript.strip()
                            result["asr_original_model"] = result.get("asr_model")
                            result["asr_model"] = "gemini-audio-transcription"
                            result["asr_provider"] = "gemini_api"
                            result["asr_fallback"] = True
                            result["asr_fallback_reason"] = "whisper_quality_below_0.60"
                            result["asr_quality_status"] = "gemini_audio_fallback_unscored"
                            result["asr_quality_note"] = (
                                "Gemini transcribed the original audio because Whisper's "
                                "quality heuristic was below the fallback threshold. "
                                "The Gemini transcript is not independently confidence-scored."
                            )
                            from models.language_spans import annotate_language_spans
                            annotation = annotate_language_spans(
                                result["transcript"], result.get("language")
                            )
                            result.update(annotation)
                            result["code_switch_detected"] = bool(
                                annotation.get("code_switch_detected")
                                or annotation.get("mixed_script_detected")
                            )
                            print(
                                "[Deployment] Replaced low-quality Whisper hypothesis "
                                "with Gemini audio transcript; original retained in diagnostics."
                            )
                    except Exception as transcription_error:
                        result["asr_audio_fallback_error"] = (
                            type(transcription_error).__name__ + ": " + str(transcription_error)
                        )
                        print("[Deployment] Gemini audio fallback unavailable: "
                              + result["asr_audio_fallback_error"])
                result["model_init_time_sec"] = model_init_time
                st.session_state["speechmind_result"] = result
                st.session_state["speechmind_model_init_time"] = model_init_time
        except Exception as error:
            st.error(f"Analysis failed: {error}")
            st.exception(error)
            st.session_state.pop("speechmind_result", None)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    result = st.session_state.get("speechmind_result")
    if result is not None:
        transcript_key = hashlib.sha256(
            (str(result.get("transcript", "")) + "|" + str(result.get("language", "unknown"))).encode("utf-8")
        ).hexdigest()
        if st.session_state.get("speechmind_insights_digest") == transcript_key:
            insights = st.session_state["speechmind_insights"]
        else:
            with st.spinner("Running conversation intelligence analysis..."):
                insights = analyze_conversation(
                    result.get("transcript", ""),
                    result.get("meaning_en") if result.get("meaning_en_available") else None,
                    language=result.get("language", "unknown"),
                )
            st.session_state["speechmind_insights"] = insights
            st.session_state["speechmind_insights_digest"] = transcript_key
        # Preserve IndicTrans2 when it succeeded; use local LLM translation only
        # when the translation branch was skipped or unavailable.
        if not result.get("meaning_en") and insights.get("english_meaning"):
            result["meaning_en"] = insights["english_meaning"]
            result["meaning_en_available"] = True
            provider = (insights.get("llm") or {}).get("provider")
            if provider == "gemini_api":
                result["meaning_en_method"] = "gemini_api"
                result["meaning_en_source"] = "Gemini API"
                result["meaning_en_note"] = "Generated by Gemini API. Verify against the original speech."
            else:
                result["meaning_en_method"] = "local_ollama_llm"
                result["meaning_en_source"] = "local_ollama"
                result["meaning_en_note"] = "Generated locally by the open-source Ollama model. Verify against the original speech."
        llm_info = insights.get("llm") or {}
        if llm_info.get("provider") in {"ollama_local", "gemini_api"}:
            st.caption(
                f"Conversation analysis: {'Gemini API' if llm_info.get('provider') == 'gemini_api' else 'local AI'} ({llm_info.get('model')}) · "
                f"{llm_info.get('duration_sec', 0)}s · {'cloud API' if llm_info.get('provider') == 'gemini_api' else 'local inference'}"
            )
        else:
            st.warning(
                "AI provider is unavailable; deterministic fallback is being used. "
                + str(llm_info.get("setup_hint", ""))
            )
        st.divider()
        st.subheader("2. Call review")
        st.caption("Start with the review summary. Open the sections below when you need more detail.")

        language = str(result.get("language", "unknown")).upper()
        code_switch = "Detected" if result.get("code_switch_detected") else "Not detected"
        topic_count = len(insights.get("topics", []))
        signal_count = len(insights.get("signals", []))
        metrics = st.columns(4)
        metrics[0].metric("Primary language", language)
        metrics[1].metric("Code-switching", code_switch)
        metrics[2].metric("Topics found", topic_count)
        metrics[3].metric("Review cues", signal_count)

        case_review = insights.get("case_review") or {}
        status = str(case_review.get("suggested_status", "review_required")).replace("_", " ").title()
        case_brief = insights.get("case_brief") or {}
        st.markdown("#### Executive case brief")
        with st.container(border=True):
            st.markdown(f"**Issue category:** {case_brief.get('primary_category', 'Needs classification')}")
            st.markdown("**What the customer reported**")
            st.write(case_brief.get("issue_summary") or "No usable transcript is available.")
            st.caption("Transcript-derived preview. The executive must verify it against the recording.")
            details = case_brief.get("reported_details") or {}
            if details:
                st.markdown("**Details found in the transcript**")
                for label, item in details.items():
                    st.write(f"- {label.replace('_', ' ').title()}: {item.get('value')}")
            st.markdown("**Information to confirm with the customer**")
            for question in (insights.get("case_review") or {}).get("suggested_questions", []):
                st.write(f"- {question}")
            st.markdown("**Recommended handling**")
            st.write(case_review.get("suggested_next_step", "Review the transcript and verify the customer's concern."))
            st.caption("Suggested status: " + status + ". This is a review aid, not a verified case outcome.")

        st.markdown("#### Findings")
        if insights.get("topics"):
            topic_cols = st.columns(min(3, len(insights["topics"])))
            for index, topic in enumerate(insights["topics"]):
                with topic_cols[index % len(topic_cols)]:
                    with st.container(border=True):
                        st.markdown(f"**{topic['topic']}**")
                        if topic.get("summary"):
                            st.write(topic["summary"])
                        if topic.get("matched_terms"):
                            st.caption("Matched terms: " + ", ".join(topic.get("matched_terms", [])))
                        st.caption("Evidence source: " + topic.get("evidence_source", "original_transcript").replace("_", " "))
                        for evidence in topic.get("evidence", []):
                            st.write("“" + evidence.get("text", "") + "”")
        else:
            st.caption("No configured topic keywords were detected. This does not mean the call has no issue.")

        if insights.get("signals"):
            for signal in insights["signals"]:
                with st.expander(signal.get("label", "Review signal"), expanded=True):
                    st.write(signal.get("description", ""))
                    if signal.get("evidence_sentences"):
                        st.markdown("**Transcript evidence**")
                        for evidence in signal["evidence_sentences"]:
                            st.write("“" + evidence.get("text", "") + "”")
                    if signal.get("evidence"):
                        st.caption("Matched cues: " + ", ".join(signal["evidence"]))
        else:
            st.info("No configured review cues were detected. This is not proof that the call is resolved or requires no action.")

        st.caption("Verify every finding against the original audio before recording a case outcome.")

        # Keep the detailed model output available without overwhelming the main review screen.
        with st.expander("Transcript and English meaning", expanded=False):
            st.subheader("Transcript (original speech)")
            left, right = st.columns(2)
            with left:
                st.markdown("**Original transcript**")
                st.write(result.get("transcript") or "No transcript was produced.")
                if result.get("asr_fallback") and result.get("asr_whisper_transcript"):
                    st.caption("Original Whisper hypothesis (retained for comparison)")
                    st.write(result["asr_whisper_transcript"])
                    st.info("The displayed transcript was produced by Gemini from the original audio because Whisper's quality heuristic was low. Gemini's transcript is not independently confidence-scored; verify it against the recording.")
                if result.get("asr_audio_fallback_error"):
                    st.warning("Gemini audio transcription fallback failed: " + str(result["asr_audio_fallback_error"]))
                if result.get("asr_error"):
                    st.warning("Speech recognition issue: " + str(result["asr_error"]))
            with right:
                st.markdown("**English meaning**")
                st.write(result.get("meaning_en") or "English meaning is unavailable.")
                if result.get("meaning_en_note"):
                    st.caption(str(result["meaning_en_note"]))
            if result.get("asr_quality_status") == "unavailable":
                st.warning(result.get("asr_quality_note") or "Transcript quality score is unavailable. Check the original transcript.")
            elif result.get("asr_quality_score") is not None:
                st.caption(
                    f"Transcript quality estimate: {float(result['asr_quality_score']) * 100:.1f}% "
                                       "(uncalibrated heuristic, not a probability of correctness)."
                )

        with st.expander("Language and code-switch details", expanded=False):
            st.caption("Language spans are transcript character ranges, not audio timestamps.")
            spans = result.get("language_spans") or []
            if spans:
                st.dataframe([
                    {
                        "Text": span.get("text", ""),
                        "Language": span.get("language", "und"),
                        "Script": span.get("script", "Other"),
                        "Character range": f"{span.get('char_start', 0)}–{span.get('char_end', 0)}",
                    }
                    for span in spans
                ], use_container_width=True, hide_index=True)
            else:
                st.caption("No language-bearing spans were detected.")

        with st.expander("Audio quality", expanded=False):
            quality = result.get("audio_quality") or {}
            if quality:
                q1, q2, q3 = st.columns(3)
                q1.metric("Duration", f"{quality.get('duration_seconds', 0):.1f} s")
                q2.metric("Peak level", f"{quality.get('peak_amplitude', 0):.3f}")
                q3.metric("Clipped samples", f"{100 * float(quality.get('clipped_sample_ratio', 0)):.2f}%")
                st.caption(
                    f"Near-silence sample share: {100 * float(quality.get('near_silence_sample_ratio', 0)):.1f}%. "
                    "Measurements only; no trimming, denoising or automatic gain change."
                )
                if quality.get("status") == "unusable":
                    st.error("The recording is effectively silent. Upload or record it again.")
                elif quality.get("flags"):
                    st.warning("Audio needs review: " + ", ".join(quality["flags"]))
            else:
                st.caption("Audio quality measurements are unavailable.")

        with st.expander("Experimental emotion signals (optional)", expanded=False):
            st.warning(
                "These model outputs are experimental and are not a reliable measure of a person's true emotional state. "
                "They are not used to determine support-case status."
            )
            emotion_cols = st.columns(2)
            emotion_cols[0].metric("Text model signal", str(result.get("text_emotion", "unavailable")).title())
            emotion_cols[1].metric("Voice model signal", str(result.get("voice_emotion", "unavailable")).title())
            if result.get("branch_comparison"):
                st.write(result["branch_comparison"])
            st.caption("The voice model has not been validated on spontaneous Indian/Hinglish support calls.")

        with st.expander("Technical diagnostics", expanded=False):
            st.caption(
                f"Model initialization: {result.get('model_init_time_sec', 0)} seconds. "
                f"Total processing: {result.get('processing_time_sec', 0)} seconds."
            )
            st.json({
                "stage_timings_sec": result.get("timings_sec", {}),
                "asr_model": result.get("asr_model"),
                "asr_provider": result.get("asr_provider"),
                "asr_quality_status": result.get("asr_quality_status"),
                "asr_quality_score": result.get("asr_quality_score"),
                "asr_quality_note": result.get("asr_quality_note"),
                "asr_fallback": result.get("asr_fallback", False),
                "asr_fallback_reason": result.get("asr_fallback_reason"),
                "asr_original_model": result.get("asr_original_model"),
                "asr_audio_fallback_error": result.get("asr_audio_fallback_error"),
                "translator_provider": result.get("translator_provider"),
                "meaning_en_method": result.get("meaning_en_method"),
                "torch_runtime": result.get("torch_runtime"),
                "uncertainty_reasons": result.get("uncertainty_reasons"),
                "language_source": result.get("language_source"),
                "audio_quality": result.get("audio_quality"),
            })

        export_payload = {
            "product": "SpeechMind",
            "report_type": "customer_support_call_review",
            "analysis": result,
            "conversation_insights": insights,
            "review_status": "pending_human_verification",
        }
        st.divider()
        st.download_button(
            "Download complete analysis (JSON)",
            data=json.dumps(export_payload, ensure_ascii=False, indent=2, default=str),
            file_name="speechmind-analysis.json",
            mime="application/json",
            key=f"speechmind_download_{audio_digest}",
        )
else:
    st.info("Add an audio recording above to begin a call review.")
    st.markdown("**How it works**")
    st.markdown(
        "1. Add a support-call recording.\n"
        "2. SpeechMind transcribes the call and generates an English meaning where supported.\n"
        "3. Review detected topics, evidence and follow-up cues.\n"
        "4. A human reviewer verifies the findings and decides what action to take."
    )
