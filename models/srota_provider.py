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
    CONVERSATIONAL_MODEL_NAME = "moorlee/qwen3-asr-0.6b-hinglish-hiacc-v1"
    MAX_CLIP_SECONDS = 30.0

    def __init__(
        self,
        project_root: str | Path | None = None,
        startup_timeout: float = 300.0,
        request_timeout: float = 180.0,
        python_executable: str | Path | None = None,
        worker_script: str | Path | None = None,
        model_name: str | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.worker_script = Path(
            worker_script or self.project_root / "scripts" / "srota_asr_worker.py"
        )
        if python_executable is not None:
            self.python_executable = Path(python_executable)
        elif os.name == "nt":
            self.python_executable = (
                self.project_root / ".venv_srota" / "Scripts" / "python.exe"
            )
        else:
            self.python_executable = (
                self.project_root / ".venv_srota" / "bin" / "python"
            )
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.model_name = str(model_name or self.MODEL_NAME).strip()
        if not self.model_name:
            raise ValueError("Srota model_name must be a non-empty Hugging Face model ID.")
        self._process: subprocess.Popen[str] | None = None
        self._stderr_handle = None
        self.stderr_log_path = self.project_root / "logs" / "srota_worker_stderr.log"
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        atexit.register(self.close)

    @property
    def available(self) -> bool:
        return self.python_executable.is_file() and self.worker_script.is_file()

    @staticmethod
    def _read_stdout(
        process: subprocess.Popen[str], responses: queue.Queue[str | None]
    ) -> None:
        if process.stdout is None:
            responses.put(None)
            return
        try:
            for line in process.stdout:
                responses.put(line.rstrip())
        finally:
            responses.put(None)

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
        if self._process is not None:
            self.close()
        if not self.available:
            raise FileNotFoundError(
                "Srota environment is unavailable. Expected .venv_srota and "
                "scripts/srota_asr_worker.py."
            )

        self._responses = queue.Queue()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        worker_env = os.environ.copy()
        worker_env["PYTHONIOENCODING"] = "utf-8:backslashreplace"
        worker_env["PYTHONUTF8"] = "1"
        self.stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_handle = self.stderr_log_path.open(
            "a", encoding="utf-8", errors="replace"
        )
        try:
            self._process = subprocess.Popen(
                [str(self.python_executable), str(self.worker_script)],
                cwd=str(self.project_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_handle,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
                env=worker_env,
            )
        except Exception:
            self._stderr_handle.close()
            self._stderr_handle = None
            raise
        response_queue = self._responses
        threading.Thread(
            target=self._read_stdout,
            args=(self._process, response_queue),
            name="speechmind-srota-stdout",
            daemon=True,
        ).start()

        try:
            ready = self._receive(self.startup_timeout)
            if not ready.get("ready"):
                raise RuntimeError(ready.get("error") or "Srota worker did not become ready.")
        except Exception:
            # Do not leave a half-started worker or an open stderr handle behind
            # when initialization fails before the request loop begins.
            self.close()
            raise

    def _request(self, payload: dict, timeout: float | None = None) -> dict:
        with self._lock:
            self._start()
            process = self._process
            if process is None or process.stdin is None:
                raise RuntimeError("Isolated model worker stdin is unavailable.")
            process.stdin.write(
                json.dumps(payload, ensure_ascii=True) + "\n"
            )
            process.stdin.flush()
            effective_timeout = timeout or self.request_timeout
            try:
                response = self._receive(effective_timeout)
            except TimeoutError as error:
                # A timed-out worker may still emit a late JSON response. Kill
                # it and discard its response queue so that it cannot be
                # mistaken for the next request's result.
                self.close()
                raise TimeoutError(
                    f"Srota request timed out after {effective_timeout:.0f}s. "
                    f"Worker stopped; stderr log: {self.stderr_log_path}"
                ) from error

        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "Isolated model worker request failed.")
        return response

    def transcribe_file(
        self, audio_path: str | Path, context: str | None = None
    ) -> dict:
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Follow the model card's native inference path: no injected instruction
        # prompt. Context is reserved for actual user-supplied names/terms only.
        final_context = str(context or "").strip()

        response = self._request({
            "action": "transcribe",
            "audio_path": str(path),
            "context": final_context,
            "model_name": self.model_name,
        })
        transcript = str(response.get("transcript") or "").strip()
        if not transcript:
            raise RuntimeError("Srota returned an empty transcript.")
        return {
            "transcript": transcript,
            "asr_model": str(response.get("model") or self.model_name),
            "inference_seconds": response.get("inference_seconds"),
            "worker_initialization_seconds": response.get(
                "worker_initialization_seconds"
            ),
            "worker_request_seconds": response.get("worker_request_seconds"),
            "worker_response": response,
        }

    def translate_to_english(self, text: str, src_lang: str) -> dict:
        """Translate text using the lazily loaded IndicTrans2 model."""
        if not text or not text.strip():
            raise ValueError("Translation input is empty.")
        if not src_lang or not src_lang.strip():
            raise ValueError("Translation source language is missing.")
        response = self._request({
            "action": "translate",
            "text": text,
            "src_lang": src_lang,
        }, timeout=max(self.request_timeout, 600.0))
        translated = str(response.get("translated_text") or "").strip()
        if not translated:
            raise RuntimeError("IndicTrans2 returned an empty translation.")
        return {
            "translated_text": translated,
            "model": response.get(
                "model", "ai4bharat/indictrans2-indic-en-dist-200M"
            ),
            "inference_seconds": response.get("inference_seconds"),
            "worker_response": response,
        }

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        context: str | None = None,
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
            return self.transcribe_file(temp_path, context=context)
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
                process.wait(timeout=2)
        finally:
            # Close the parent-side pipe wrappers as well as the child process.
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
            if self._stderr_handle is not None:
                if not self._stderr_handle.closed:
                    self._stderr_handle.close()
                self._stderr_handle = None
