"""Persistent local model worker for Srota ASR and IndicTrans2 translation.

Runs inside .venv_srota so the main SpeechMind environment keeps its
Transformers 4.46.3 pin. Models are loaded lazily on first request.
"""
from __future__ import annotations

import json
import os
import sys
import time
from importlib.util import find_spec
from contextlib import redirect_stdout
from io import StringIO

# This process communicates over pipes, not a Windows console. Force UTF-8
# for input and diagnostics, and emit ASCII-escaped JSON on the protocol pipe.
for stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

_srota_model = None
_srota_model_name = None
_srota_device = None
_translation_components = None
_torch = None
_srota_torch_threads = 4


def configured_cpu_threads(environ=None) -> int:
    """Read a bounded CPU thread count for the isolated Srota worker."""
    values = os.environ if environ is None else environ
    try:
        requested = int(values.get("SPEECHMIND_SROTA_TORCH_THREADS", "4"))
    except (TypeError, ValueError):
        requested = 4
    return max(1, min(requested, 16))


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def get_torch():
    global _torch, _srota_torch_threads
    if _torch is None:
        import torch
        _torch = torch
        _srota_torch_threads = configured_cpu_threads()
        torch.set_num_threads(_srota_torch_threads)
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


def get_srota_model(model_name: str):
    global _srota_model, _srota_model_name, _srota_device
    requested_name = str(model_name or "moorlee/qwen3-asr-0.6b-hinglish").strip()
    if not requested_name:
        raise ValueError("Srota request is missing a valid model_name.")
    if _srota_model is not None and requested_name != _srota_model_name:
        raise RuntimeError(
            "This persistent worker already loaded "
            f"{_srota_model_name}; start a new provider worker to load {requested_name}."
        )
    if _srota_model is None:
        torch = get_torch()
        preference = os.environ.get("SPEECHMIND_SROTA_DEVICE", "auto").strip().lower()
        if preference not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                "SPEECHMIND_SROTA_DEVICE must be one of: auto, cpu, cuda."
            )
        cuda_available = bool(torch.cuda.is_available())
        if preference == "cuda" and not cuda_available:
            raise RuntimeError(
                "SPEECHMIND_SROTA_DEVICE=cuda was requested, but CUDA is unavailable."
            )
        use_cuda = cuda_available and preference != "cpu"
        device_map = "cuda:0" if use_cuda else "cpu"
        dtype = (
            torch.bfloat16
            if use_cuda and torch.cuda.is_bf16_supported()
            else torch.float16 if use_cuda else torch.float32
        )
        model_kwargs = {
            "dtype": dtype,
            "device_map": device_map,
            "max_inference_batch_size": 1,
            "max_new_tokens": 512,
        }
        if use_cuda:
            try:
                if find_spec("flash_attn") is not None:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
            except (ImportError, ValueError):
                pass

        with redirect_stdout(StringIO()):
            from qwen_asr import Qwen3ASRModel
            try:
                _srota_model = Qwen3ASRModel.from_pretrained(
                    requested_name, **model_kwargs
                )
            except Exception:
                # In auto mode, a GPU load failure (e.g. unsupported attention
                # kernels or insufficient VRAM) falls back to CPU.
                if not use_cuda or preference != "auto":
                    raise
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                _srota_model = Qwen3ASRModel.from_pretrained(
                    requested_name,
                    dtype=torch.float32,
                    device_map="cpu",
                    max_inference_batch_size=1,
                    max_new_tokens=512,
                )
                device_map = "cpu"
        _srota_model_name = requested_name
        _srota_device = device_map
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
                model_name = str(
                    request.get("model_name") or "moorlee/qwen3-asr-0.6b-hinglish"
                ).strip()
                model = get_srota_model(model_name)
                context = str(request.get("context") or "").strip()
                with redirect_stdout(StringIO()), torch.inference_mode():
                    result = model.transcribe(
                        audio=str(audio_path), language=None, context=context
                    )
                transcript = extract_text(result)
                if not transcript:
                    raise RuntimeError("Srota returned an empty transcript.")
                emit({
                    "ok": True,
                    "transcript": transcript,
                    "inference_seconds": round(time.perf_counter() - started, 3),
                    "model": model_name,
                    "device": _srota_device,
                    "torch_threads": _srota_torch_threads,
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
