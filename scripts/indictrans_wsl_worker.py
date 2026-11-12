"""Persistent Linux/WSL worker for IndicTrans2 Indic-to-English inference."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
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
        get_torch()
        # Keep third-party/model initialization messages off the JSONL stdout
        # channel. AI4Bharat's Transformers example uses eager attention when
        # Flash Attention 2 is not installed; this CPU worker does the same.
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
                MODEL_NAME,
                trust_remote_code=True,
                attn_implementation="eager",
            ).to("cpu")
            model.eval()
            processor = IndicProcessor(inference=True)
        _components = tokenizer, model, processor
    return _components




def generate_translation_with_cache(model, inputs, torch, environ=None):
    """Prefer decoder KV caching; retry without it only for cache incompatibility."""
    environment = os.environ if environ is None else environ
    cache_setting = str(
        environment.get("SPEECHMIND_INDIC_TRANS_USE_CACHE", "1")
    ).strip().lower()
    requested_cache = cache_setting not in {"0", "false", "no", "off"}

    def generate(use_cache):
        with torch.inference_mode():
            return model.generate(
                **inputs,
                use_cache=use_cache,
                min_length=0,
                max_length=256,
                num_beams=4,
                num_return_sequences=1,
            )

    try:
        return generate(requested_cache), requested_cache
    except Exception as error:
        message = str(error).casefold()
        cache_incompatibility = any(
            marker in message
            for marker in ("use_cache", "past_key_values", "cache_position", "cache")
        )
        if not requested_cache or not cache_incompatibility:
            raise
        # Some model/Transformers combinations expose generate() without a
        # compatible cache implementation. Preserve translation availability.
        return generate(False), False


def translate_to_english(text: str, src_lang: str) -> tuple[str, bool]:
    stage = "input validation"
    try:
        if not text.strip():
            raise ValueError("Translation input is empty.")
        if not src_lang.strip():
            raise ValueError("Translation source language is missing.")

        stage = "loading IndicTrans2 model/tokenizer"
        torch = get_torch()
        tokenizer, model, processor = get_components()

        stage = "IndicProcessor preprocessing"
        prepared = processor.preprocess_batch(
            [text], src_lang=src_lang, tgt_lang="eng_Latn"
        )
        if not prepared or not str(prepared[0]).strip():
            raise RuntimeError("IndicProcessor returned an empty prepared sentence.")

        stage = "tokenization"
        inputs = tokenizer(
            prepared,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=256,
            return_attention_mask=True,
        )
        if "input_ids" not in inputs or inputs["input_ids"] is None:
            raise RuntimeError("Tokenizer did not return input_ids.")
        if "attention_mask" not in inputs or inputs["attention_mask"] is None:
            raise RuntimeError("Tokenizer did not return attention_mask.")
        inputs = inputs.to("cpu")

        stage = "sequence generation"
        generated, cache_used = generate_translation_with_cache(
            model, inputs, torch
        )

        stage = "token decoding"
        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        if not decoded:
            raise RuntimeError("Tokenizer decoded no generated sequences.")

        stage = "IndicProcessor postprocessing"
        translated = processor.postprocess_batch(decoded, lang="eng_Latn")
        result = str(translated[0]).strip() if translated else ""
        if not result:
            raise RuntimeError("IndicTrans2 returned an empty English translation.")
        return result, cache_used
    except Exception as error:
        detail = traceback.format_exc()
        raise RuntimeError(
            f"IndicTrans2 stage '{stage}' failed: "
            f"{type(error).__name__}: {error}\n{detail}"
        ) from error


def main() -> int:
    emit({"ready": True, "worker": "speechmind-indictrans-wsl"})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            action = str(request.get("action") or "").lower()
            if action != "translate":
                raise ValueError(f"Unknown worker action: {action}")
            started = time.perf_counter()
            translated, cache_used = translate_to_english(
                str(request.get("text") or ""),
                str(request.get("src_lang") or ""),
            )
            emit({
                "ok": True,
                "translated_text": translated,
                "inference_seconds": round(time.perf_counter() - started, 3),
                "model": MODEL_NAME,
                "cache_used": cache_used,
            })
        except Exception as error:
            emit({"ok": False, "error": f"{type(error).__name__}: {error}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
