"""Hugging Face Gradio entry point for the SpeechMind hosted demo.

The local Streamlit application and production pipeline remain unchanged.
This hosted adapter intentionally disables the isolated Srota/IndicTrans2
workers: those rely on a separate Python environment that the free Gradio
Space does not provide. Hindi uses IndicConformer; English/Hinglish uses
Whisper, with the routing limitation disclosed in the UI.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

import gradio as gr

from models.deployment_checkpoints import ensure_deployment_checkpoints

_ROOT = Path(__file__).resolve().parent
_PIPELINE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_pipeline():
    """Load model checkpoints before constructing one reusable pipeline."""
    ensure_deployment_checkpoints()
    from pipeline import SpeechMindPipeline

    pipeline = SpeechMindPipeline(load_translator=False)
    # Srota requires a separate .venv_srota process. Do not start that worker
    # in a standard Gradio Space; Whisper remains the Hinglish fallback.
    if pipeline.asr is not None:
        pipeline.asr._try_srota_code_switch = lambda *args, **kwargs: None
    return pipeline


def _format_emotion(result: dict) -> str:
    keys = (
        "text_emotion", "voice_emotion", "fused_emotion",
        "combined_emotion", "emotion_fusion",
    )
    found = {key: result.get(key) for key in keys if key in result}
    return json.dumps(found or {
        "text_emotion": result.get("text_emotion_result"),
        "voice_emotion": result.get("voice_emotion_result"),
        "fusion": result.get("fusion"),
    }, ensure_ascii=False, indent=2, default=str)


def analyze_audio(audio_path: str | None):
    if not audio_path:
        raise gr.Error("Upload or record an audio clip first.")

    path = Path(audio_path)
    if not path.is_file():
        raise gr.Error("The uploaded audio file is no longer available.")

    try:
        # Serialize CPU-heavy model inference to avoid concurrent memory spikes
        # on the free shared Space hardware.
        with _INFERENCE_LOCK:
            pipeline = get_pipeline()
            result = pipeline.run(
                str(path),
                include_emotion=True,
                include_translation=False,
            )
    except Exception as error:
        raise gr.Error(f"SpeechMind inference failed: {type(error).__name__}: {error}") from error

    transcript = str(result.get("transcript") or "")
    language = str(result.get("language") or "unknown")
    if not transcript:
        transcript = "No transcript was produced. Check the audio and retry."

    # Gemini is optional. If unavailable, use the deterministic evidence-based
    # rules already present in the project; never fabricate an English meaning.
    try:
        from local_llm import analyze_with_local_llm
        insights = analyze_with_local_llm(transcript, language=language)
        insight_source = insights.get("analysis_source", "Gemini / configured provider")
    except Exception as provider_error:
        try:
            from conversation_insights import analyze_conversation_text
            insights = analyze_conversation_text(transcript)
            insight_source = "Deterministic fallback"
            insights["provider_note"] = (
                f"LLM unavailable: {type(provider_error).__name__}: {provider_error}"
            )
        except Exception as fallback_error:
            insights = {
                "analysis_source": "unavailable",
                "error": f"{type(fallback_error).__name__}: {fallback_error}",
            }
            insight_source = "Unavailable"

    limitations = (
        "Hosted routing: Whisper for English/Hinglish; IndicConformer for "
        "Hindi/other supported Indic languages. Srota is disabled because "
        "its isolated worker environment is not part of this free Gradio Space. "
        "IndicTrans2 translation is disabled here to avoid a second incompatible "
        "worker runtime. Gemini may provide an English meaning when configured; "
        "otherwise the deterministic fallback does not claim to translate."
    )

    metadata = {
        "language": language,
        "language_source": result.get("language_source"),
        "asr_model": result.get("asr_model"),
        "asr_fallback": result.get("asr_fallback"),
        "translation": "Not run in hosted Gradio profile",
        "conversation_analysis_source": insight_source,
        "timings_sec": result.get("timings_sec", result.get("timings")),
        "audio_quality": result.get("audio_quality"),
        "limitations": limitations,
    }

    return (
        transcript,
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        _format_emotion(result),
        json.dumps(insights, ensure_ascii=False, indent=2, default=str),
    )


with gr.Blocks(title="SpeechMind", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# SpeechMind\n"
        "### Multilingual speech, meaning and emotion recognition\n"
        "Upload a short Hindi, Hinglish or English recording. Processing runs "
        "on shared hosted hardware and may take several minutes."
    )
    gr.Markdown(
        "**Privacy:** This is a public demo. Do not upload private, confidential, "
        "medical or sensitive recordings. Audio is processed for inference; "
        "Gemini may receive the transcript when its API key is configured."
    )
    audio = gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="Speech recording",
    )
    run = gr.Button("Analyze recording", variant="primary")
    transcript = gr.Textbox(label="Original transcript", lines=6)
    with gr.Row():
        metadata = gr.Code(label="Language, routing and runtime", language="json")
        emotion = gr.Code(label="Emotion analysis", language="json")
    insights = gr.Code(label="Conversation insights", language="json")
    run.click(
        fn=analyze_audio,
        inputs=[audio],
        outputs=[transcript, metadata, emotion, insights],
    )
    gr.Markdown(
        "SpeechMind predictions are experimental. Emotion scores are not "
        "calibrated probabilities or diagnoses. Review the transcript and "
        "evidence before relying on any interpretation."
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
