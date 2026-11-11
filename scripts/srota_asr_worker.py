"""Long-lived Srota inference worker. Keep stdout reserved for JSONL protocol."""
from __future__ import annotations

import json
import sys
import time
from contextlib import redirect_stdout


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


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


def main() -> int:
    try:
        # qwen-asr writes progress messages to stdout; keep those out of the
        # machine-readable worker protocol.
        with redirect_stdout(sys.stderr):
            import torch
            from qwen_asr import Qwen3ASRModel

            torch.set_num_threads(4)
            torch.set_num_interop_threads(2)
            model = Qwen3ASRModel.from_pretrained(
                "moorlee/qwen3-asr-0.6b-hinglish",
                dtype=torch.float32,
                device_map="cpu",
                max_inference_batch_size=1,
                max_new_tokens=512,
            )
        emit({"ready": True, "model": "moorlee/qwen3-asr-0.6b-hinglish"})
    except Exception as error:
        emit({
            "ready": False,
            "error": f"{type(error).__name__}: {error}",
        })
        return 1

    for line in sys.stdin:
        try:
            request = json.loads(line)
            audio_path = request.get("audio_path")
            if not audio_path:
                raise ValueError("Request is missing audio_path.")
            started = time.perf_counter()
            with redirect_stdout(sys.stderr):
                result = model.transcribe(audio=str(audio_path), language=None)
            transcript = extract_text(result)
            if not transcript:
                raise RuntimeError("Srota returned an empty transcript.")
            emit({
                "ok": True,
                "transcript": transcript,
                "inference_seconds": round(time.perf_counter() - started, 3),
            })
        except Exception as error:
            emit({"ok": False, "error": f"{type(error).__name__}: {error}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
