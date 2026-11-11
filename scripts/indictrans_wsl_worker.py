"""Persistent Linux/WSL worker for IndicTrans2 Indic-to-English inference."""
from __future__ import annotations

import json
import sys
import time
from contextlib import redirect_stdout
from io import StringIO

for stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"
_torch = None
_components = None


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


def get_components():
    global _components
    if _components is None:
        torch = get_torch()
        with redirect_stdout(StringIO()):
            try:
                    from IndicTransToolkit.processor import IndicProcessor
            except ImportError:
                from IndicTransToolkit import IndicProcessor
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, trust_remote_code=True
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME, trust_remote_code=True
            ).to("cpu")
            model.eval()
            processor = IndicProcessor(inference=True)
        _components = tokenizer, model, processor
    return _components


def translate_to_english(text: str, src_lang: str) -> str:
    if not text.strip():
        raise ValueError("Translation input is empty.")
    if not src_lang.strip():
        raise ValueError("Translation source language is missing.")
    torch = get_torch()
    tokenizer, model, processor = get_components()
    prepared = processor.preprocess_batch(
        [text], src_lang=src_lang, tgt_lang="eng_Latn"
    )
    inputs = tokenizer(
        prepared,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to("cpu")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            num_return_sequences=1,
        )
    decoded = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    translated = processor.postprocess_batch(decoded, lang="eng_Latn")
    result = str(translated[0]).strip() if translated else ""
    if not result:
        raise RuntimeError("IndicTrans2 returned an empty English translation.")
    return result


def main() -> int:
    emit({"ready": True, "worker": "speechmind-indictrans-wsl"})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            action = str(request.get("action") or "").lower()
            if action != "translate":
                raise ValueError(f"Unknown worker action: {action}")
            started = time.perf_counter()
            translated = translate_to_english(
                str(request.get("text") or ""),
                str(request.get("src_lang") or ""),
            )
            emit({
                "ok": True,
                "translated_text": translated,
                "inference_seconds": round(time.perf_counter() - started, 3),
                "model": MODEL_NAME,
            })
        except Exception as error:
            emit({"ok": False, "error": f"{type(error).__name__}: {error}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
