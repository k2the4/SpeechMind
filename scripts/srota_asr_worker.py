"""Persistent local model worker for Srota ASR and IndicTrans2 translation.

Runs inside .venv_srota so the main SpeechMind environment keeps its
Transformers 4.46.3 pin. Models are loaded lazily on first request.
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import redirect_stdout
from io import StringIO

# This process communicates over pipes, not a Windows console. Force UTF-8
# for input and diagnostics, and emit ASCII-escaped JSON on the protocol pipe.
for stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

_srota_model = None
_translation_components = None
_torch = None


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
        torch.set_num_threads(4)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass
    return _torch


def extract_text(result) -> str:
    if isinstance(result, (list, tuple)):
        result = result[0] if result else ""
    if isinstance(result, str):
        return result.strip()
    if hasattr(result, "text"):
        return str(result.text or "").strip()
    if isinstance(result, dict):
        return str(result.get("text") or result.get("transcription") or "").strip()
    return str(result).strip()


def get_srota_model():
    global _srota_model
    if _srota_model is None:
        torch = get_torch()
        with redirect_stdout(StringIO()):
            from qwen_asr import Qwen3ASRModel
            _srota_model = Qwen3ASRModel.from_pretrained(
                "moorlee/qwen3-asr-0.6b-hinglish",
                dtype=torch.float32,
                device_map="cpu",
                max_inference_batch_size=1,
                max_new_tokens=512,
            )
    return _srota_model


def get_translation_components():
    global _translation_components
    if _translation_components is None:
        torch = get_torch()
        with redirect_stdout(StringIO()):
            try:
                from IndicTransToolkit.processor import IndicProcessor
            except ImportError:
                from IndicTransToolkit import IndicProcessor

            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, trust_remote_code=True
            ).to("cpu")
            model.eval()
            processor = IndicProcessor(inference=True)
        _translation_components = (tokenizer, model, processor)
    return _translation_components


def translate_to_english(text: str, src_lang: str) -> str:
    if not text.strip():
        raise ValueError("Translation input is empty.")
    if not src_lang.strip():
        raise ValueError("Translation source language is missing.")
    torch = get_torch()
    tokenizer, model, processor = get_translation_components()
    batch = processor.preprocess_batch(
        [text], src_lang=src_lang, tgt_lang="eng_Latn"
    )
    inputs = tokenizer(
        batch, return_tensors="pt", padding=True, truncation=True
    ).to("cpu")
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_length=256, num_beams=4
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    translated = processor.postprocess_batch(decoded, lang="eng_Latn")
    result = str(translated[0]).strip() if translated else ""
    if not result:
        raise RuntimeError("IndicTrans2 returned an empty English translation.")
    return result


def main() -> int:
    emit({"ready": True, "worker": "speechmind-isolated-model-worker"})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            action = str(request.get("action") or "transcribe").lower()

            if action == "transcribe":
                audio_path = request.get("audio_path")
                if not audio_path:
                    raise ValueError("Transcription request is missing audio_path.")
                started = time.perf_counter()
                torch = get_torch()
                model = get_srota_model()
                with redirect_stdout(StringIO()), torch.inference_mode():
                    result = model.transcribe(
                        audio=str(audio_path), language=None
                    )
                transcript = extract_text(result)
                if not transcript:
                    raise RuntimeError("Srota returned an empty transcript.")
                emit({
                    "ok": True,
                    "transcript": transcript,
                    "inference_seconds": round(time.perf_counter() - started, 3),
                    "model": "moorlee/qwen3-asr-0.6b-hinglish",
                })

            elif action == "translate":
                started = time.perf_counter()
                text = str(request.get("text") or "")
                src_lang = str(request.get("src_lang") or "")
                translated = translate_to_english(text, src_lang)
                emit({
                    "ok": True,
                    "translated_text": translated,
                    "inference_seconds": round(time.perf_counter() - started, 3),
                    "model": "ai4bharat/indictrans2-indic-en-dist-200M",
                })
            else:
                raise ValueError(f"Unknown worker action: {action}")

        except Exception as error:
            emit({"ok": False, "error": f"{type(error).__name__}: {error}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
