"""Authenticated HTTP service for the isolated SpeechMind Srota ASR worker."""
from __future__ import annotations

import hmac
import os
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel
from qwen_asr import Qwen3ASRModel

app = FastAPI(title="SpeechMind Srota Worker", version="1.0.0")
_model = None
_model_name = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_translation_components = None
_translation_lock = threading.Lock()
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def _authorized(authorization: str | None) -> bool:
    expected = os.environ.get("SPEECHMIND_SROTA_API_TOKEN", "")
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    return bool(expected and supplied and hmac.compare_digest(expected, supplied))


def _get_model(model_name: str):
    global _model, _model_name
    with _model_lock:
        if _model is not None:
            if model_name != _model_name:
                raise HTTPException(400, "This worker is already loaded with a different model.")
            return _model
        import torch
        torch.set_num_threads(max(1, min(int(os.environ.get("SPEECHMIND_SROTA_TORCH_THREADS", "4")), 16)))
        started = time.perf_counter()
        _model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="cpu",
            max_inference_batch_size=1,
            max_new_tokens=512,
        )
        _model_name = model_name
        _model_init_seconds = time.perf_counter() - started
        return _model


def _extract_text(result) -> str:
    if isinstance(result, (list, tuple)):
        result = result[0] if result else ""
    if isinstance(result, str):
        return result.strip()
    if hasattr(result, "text"):
        return str(result.text or "").strip()
    if isinstance(result, dict):
        return str(result.get("text") or result.get("transcription") or "").strip()
    return str(result).strip()


@app.get("/health")
def health():
    return {"ok": True, "service": "speechmind-srota", "model_loaded": _model is not None}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    context: str = Form(""),
    model_name: str = Form("moorlee/qwen3-asr-0.6b-hinglish"),
    authorization: str | None = Header(default=None),
):
    if not _authorized(authorization):
        raise HTTPException(401, "Unauthorized")
    if not audio.filename:
        raise HTTPException(400, "Audio filename is missing.")
    suffix = Path(audio.filename).suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".flac"}:
        raise HTTPException(400, "Unsupported audio format.")
    payload = await audio.read()
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio is empty or exceeds the 200 MB upload limit.")

    request_started = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
    try:
        init_started = time.perf_counter()
        model = _get_model(model_name.strip())
        initialization_seconds = time.perf_counter() - init_started
        inference_started = time.perf_counter()
        with _inference_lock:
            import torch
            with torch.inference_mode():
                result = model.transcribe(
                    audio=str(temp_path), language=None, context=str(context or "").strip()
                )
        inference_seconds = time.perf_counter() - inference_started
        transcript = _extract_text(result)
        if not transcript:
            raise HTTPException(502, "Srota returned an empty transcript.")
        return {
            "ok": True,
            "transcript": transcript,
            "model": model_name,
            "device": "cpu",
            "inference_seconds": round(inference_seconds, 3),
            "worker_initialization_seconds": round(initialization_seconds, 3),
            "worker_request_seconds": round(time.perf_counter() - request_started, 3),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(500, f"{type(error).__name__}: {error}") from error
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


class TranslationRequest(BaseModel):
    text: str
    src_lang: str


def _get_translation_components():
    global _translation_components
    with _translation_lock:
        if _translation_components is not None:
            return _translation_components
        import torch
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:
            from IndicTransToolkit import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to("cpu")
        model.eval()
        processor = IndicProcessor(inference=True)
        _translation_components = (torch, tokenizer, model, processor)
        return _translation_components


@app.post("/translate")
def translate(
    request: TranslationRequest,
    authorization: str | None = Header(default=None),
):
    if not _authorized(authorization):
        raise HTTPException(401, "Unauthorized")
    if not request.text.strip() or not request.src_lang.strip():
        raise HTTPException(400, "Translation text and source language are required.")
    try:
        torch, tokenizer, model, processor = _get_translation_components()
        started = time.perf_counter()
        batch = processor.preprocess_batch(
            [request.text], src_lang=request.src_lang, tgt_lang="eng_Latn"
        )
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to("cpu")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_length=256, num_beams=4)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translated = processor.postprocess_batch(decoded, lang="eng_Latn")
        result = str(translated[0]).strip() if translated else ""
        if not result:
            raise RuntimeError("IndicTrans2 returned an empty English translation.")
        return {
            "ok": True,
            "translated_text": result,
            "model": "ai4bharat/indictrans2-indic-en-dist-200M",
            "inference_seconds": round(time.perf_counter() - started, 3),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(500, f"{type(error).__name__}: {error}") from error
