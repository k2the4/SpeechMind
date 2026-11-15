"""Hugging Face Gradio entry point for the SpeechMind hosted demo.

The local Streamlit application and production pipeline remain unchanged.
This hosted adapter intentionally disables the isolated Srota/IndicTrans2
workers: those rely on a separate Python environment that the free Gradio
Space does not provide. Hindi uses IndicConformer; English/Hinglish uses
Whisper, with the routing limitation disclosed in the UI.
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import gradio as gr

from models.deployment_checkpoints import ensure_deployment_checkpoints

_INFERENCE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_pipeline():
    """Load private checkpoints once and reuse the pipeline across requests."""
    ensure_deployment_checkpoints()
    from pipeline import SpeechMindPipeline

    pipeline = SpeechMindPipeline(load_translator=False)
    # The Srota provider depends on a separate .venv_srota worker. In this
    # hosted profile, retain the existing Whisper fallback instead.
    if pipeline.asr is not None:
        pipeline.asr._try_srota_code_switch = lambda *args, **kwargs: None
    return pipeline


def _emotion_payload(result: dict) -> str:
    keys = (
        "text_emotion", "voice_emotion", "fused_emotion",
        "combined_emotion", "emotion_fusion", "text_emotion_result",
        "voice_emotion_result", "fusion",
    )
    payload = {key: result[key] for key in keys if key in result}
    return json.dumps(payload or {
        "overall_emotion": result.get("overall_emotion", "unavailable"),
        "note": "Emotion fields were not exposed by the pipeline result.",
    }, ensure_ascii=False, indent=2, default=str)


def analyze_audio(audio_path: str | None):
    if not audio_path:
        raise gr.Error("Upload or record an audio clip first.")
    path = Path(audio_path)
    if not path.is_file():
        raise gr.Error("The uploaded audio file is no longer available.")

    try:
        # Serialize CPU-heavy inference to reduce memory spikes on shared free
        # hardware. Translation stays disabled in this hosted profile.
        with _INFERENCE_LOCK:
            pipeline = get_pipeline()
            result = pipeline.run(
                str(path), include_emotion=True, include_translation=False
            )
    except Exception as error:
        raise gr.Error(
            f"SpeechMind inference failed: {type(error).__name__}: {error}"
        ) from error

    transcript = str(result.get("transcript") or "")
    language = str(result.get("language") or "unknown")
    if not transcript:
        transcript = "No transcript was produced. Check the audio and retry."

    # Prefer Gemini when configured. On provider failure, use the project's
    # deterministic evidence-based analyzer and clearly label the fallback.
    try:
        from local_llm import analyze_with_local_llm
        insights = analyze_with_local_llm(transcript, language=language)
        insight_source = insights.get("analysis_source", "configured LLM provider")
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

    metadata = {
        "language": language,
        "language_source": result.get("language_source"),
        "asr_model": result.get("asr_model"),
        "asr_fallback": result.get("asr_fallback"),
        "translation": "Disabled in hosted profile",
        "conversation_analysis_source": insight_source,
        "timings_sec": result.get("timings_sec", result.get("timings")),
        "audio_quality": result.get("audio_quality"),
        "hosted_routing_note": (
            "Whisper handles English-detected speech. IndicConformer handles "
            "Hindi and supported Indic-language detections. Hindi-detected "
            "mixed Hindi-English speech does not use Srota in this hosted profile. Srota and IndicTrans2 are "
            "disabled because their isolated worker environment is not included "
            "in this free Gradio Space. Gemini may provide English meaning when "
            "configured; deterministic fallback does not claim to translate."
        ),
    }

    return (
        transcript,
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        _emotion_payload(result),
        json.dumps(insights, ensure_ascii=False, indent=2, default=str),
    )


with gr.Blocks(title="SpeechMind", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# SpeechMind\n"
        "### Multilingual speech, meaning and emotion recognition\n"
        "Upload a short Hindi, Hinglish or English recording. Processing uses "
        "shared hosted hardware and may take several minutes."
    )
    gr.Markdown(
        "**Privacy:** This is a public demo. Do not upload private, confidential "
        "or sensitive recordings. When configured, Gemini receives the transcript "
        "for conversation analysis."
    )
    audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                     label="Speech recording")
    run = gr.Button("Analyze recording", variant="primary")
    transcript = gr.Textbox(label="Original transcript", lines=6)
    with gr.Row():
        metadata = gr.Code(label="Language, routing and runtime", language="json")
        emotion = gr.Code(label="Emotion analysis", language="json")
    insights = gr.Code(label="Conversation insights", language="json")
    run.click(fn=analyze_audio, inputs=audio,
              outputs=[transcript, metadata, emotion, insights])
    gr.Markdown(
        "Predictions are experimental. Emotion scores are not calibrated "
        "probabilities or diagnoses. Verify the transcript and evidence."
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
