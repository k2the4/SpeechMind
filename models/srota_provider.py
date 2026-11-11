"""Persistent bridge to the isolated Srota/qwen-asr environment.

The main SpeechMind environment keeps its pinned Transformers version. This
provider starts one long-lived worker from .venv_srota and exchanges JSONL
messages over stdin/stdout.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import threading
import atexit
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf


def has_mixed_devanagari_latin(text: str) -> bool:
    """Return true when text contains both Devanagari and Latin letters."""
    scripts = set()
    for character in str(text or ""):
        name = unicodedata.name(character, "")
        if "DEVANAGARI" in name:
            scripts.add("Devanagari")
        elif "LATIN" in name:
            scripts.add("Latin")
        if len(scripts) == 2:
            return True
    return False


class SrotaASRProvider:
    MODEL_NAME = "moorlee/qwen3-asr-0.6b-hinglish"
    MAX_CLIP_SECONDS = 30.0

    def __init__(
        self,
        project_root: str | Path | None = None,
        startup_timeout: float = 300.0,
        request_timeout: float = 180.0,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.worker_script = self.project_root / "scripts" / "srota_asr_worker.py"
        if os.name == "nt":
            self.python_executable = (
                self.project_root / ".venv_srota" / "Scripts" / "python.exe"
            )
        else:
            self.python_executable = (
                self.project_root / ".venv_srota" / "bin" / "python"
            )
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        atexit.register(self.close)

    @property
    def available(self) -> bool:
        return self.python_executable.is_file() and self.worker_script.is_file()

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            self._responses.put(None)
            return
        try:
            for line in process.stdout:
                self._responses.put(line.rstrip("\r\n"))
        finally:
            self._responses.put(None)

    def _receive(self, timeout: float) -> dict:
        try:
            line = self._responses.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError("Timed out waiting for the Srota worker.") from error
        if line is None:
            code = self._process.poll() if self._process is not None else None
            raise RuntimeError(f"Srota worker exited unexpectedly (code={code}).")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Srota worker returned invalid JSON: {line[:300]}") from error
        if not isinstance(payload, dict):
            raise RuntimeError("Srota worker returned an invalid response.")
        return payload

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if not self.available:
            raise FileNotFoundError(
                "Srota environment is unavailable. Expected .venv_srota and "
                "scripts/srota_asr_worker.py."
            )

        self._responses = queue.Queue()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            [str(self.python_executable), str(self.worker_script)],
            cwd=str(self.project_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=creation_flags,
        )
        response_queue = self._responses
        threading.Thread(
            target=self._read_stdout,
            args=(self._process, response_queue),
            name="speechmind-srota-stdout",
            daemon=True,
        ).start()

        ready = self._receive(self.startup_timeout)
        if not ready.get("ready"):
            raise RuntimeError(ready.get("error") or "Srota worker did not become ready.")

    def transcribe_file(self, audio_path: str | Path) -> dict:
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        with self._lock:
            self._start()
            process = self._process
            if process is None or process.stdin is None:
                raise RuntimeError("Srota worker stdin is unavailable.")
            process.stdin.write(json.dumps({"audio_path": str(path)}, ensure_ascii=False) + "\n")
            process.stdin.flush()
            response = self._receive(self.request_timeout)

        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "Srota inference failed.")
        transcript = str(response.get("transcript") or "").strip()
        if not transcript:
            raise RuntimeError("Srota returned an empty transcript.")
        return {
            "transcript": transcript,
            "asr_model": self.MODEL_NAME,
            "inference_seconds": response.get("inference_seconds"),
            "worker_response": response,
        }

    def transcribe_audio(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> dict:
        waveform = np.asarray(audio, dtype=np.float32).squeeze()
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Srota expects non-empty mono audio.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        duration = waveform.size / float(sample_rate)
        if duration > self.MAX_CLIP_SECONDS:
            raise ValueError(
                f"Srota currently accepts clips up to {self.MAX_CLIP_SECONDS:.0f} "
                f"seconds; received {duration:.1f} seconds."
            )
        if not np.isfinite(waveform).all():
            raise ValueError("Audio contains non-finite samples.")

        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(handle.name)
        handle.close()
        try:
            sf.write(str(temp_path), waveform, sample_rate, subtype="PCM_16")
            return self.transcribe_file(temp_path)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
