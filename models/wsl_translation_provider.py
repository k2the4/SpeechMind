"""Persistent WSL bridge for the isolated IndicTrans2 translation runtime.

The Windows app communicates with a JSONL worker running in the default WSL
Linux distribution. Translation dependencies never enter the main Windows venv
or the Srota ASR venv.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from utils.translation_cache import TranslationResultCache


class WslIndicTransProvider:
    VENV_PATH = "~/.venvs/speechmind-indictrans"
    WORKER_RELATIVE_PATH = "scripts/indictrans_wsl_worker.py"

    def __init__(
        self,
        project_root: str | Path | None = None,
        startup_timeout: float = 90.0,
        request_timeout: float = 600.0,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.worker_script = self.project_root / self.WORKER_RELATIVE_PATH
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._wsl_project_root: str | None = None
        self._translation_cache = TranslationResultCache(max_items=64)
        atexit.register(self.close)

    @property
    def available(self) -> bool:
        return (
            os.name == "nt"
            and shutil.which("wsl.exe") is not None
            and self.worker_script.is_file()
        )

    @staticmethod
    def _read_stdout(
        process: subprocess.Popen[str], responses: queue.Queue[str | None]
    ) -> None:
        if process.stdout is None:
            responses.put(None)
            return
        try:
            for line in process.stdout:
                responses.put(line.rstrip("\r\n"))
        finally:
            responses.put(None)

    def _get_wsl_project_root(self) -> str:
        if self._wsl_project_root:
            return self._wsl_project_root
        if shutil.which("wsl.exe") is None:
            raise FileNotFoundError(
                "WSL is not installed. Install an Ubuntu WSL distribution, "
                "then run SETUP_INDIC_TRANS_WSL.bat."
            )
        result = subprocess.run(
            ["wsl.exe", "--exec", "wslpath", "-a", str(self.project_root)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            detail = (result.stderr or result.stdout or "wslpath failed").strip()
            raise RuntimeError(
                "Could not map the SpeechMind folder into WSL. "
                "Open an Ubuntu WSL terminal once and complete first-time setup. "
                + detail
            )
        self._wsl_project_root = result.stdout.strip().splitlines()[-1]
        return self._wsl_project_root

    def _receive(self, timeout: float) -> dict[str, Any]:
        try:
            line = self._responses.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError(
                "Timed out waiting for the IndicTrans2 WSL worker."
            ) from error
        if line is None:
            code = self._process.poll() if self._process is not None else None
            raise RuntimeError(
                f"IndicTrans2 WSL worker exited unexpectedly (code={code}). "
                "Run SETUP_INDIC_TRANS_WSL.bat to verify its environment."
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"IndicTrans2 worker returned invalid JSON: {line[:300]}"
            ) from error
        if not isinstance(payload, dict):
            raise RuntimeError("IndicTrans2 worker returned an invalid response.")
        return payload

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if not self.available:
            raise FileNotFoundError(
                "IndicTrans2 WSL runtime is unavailable. Run "
                "SETUP_INDIC_TRANS_WSL.bat from the SpeechMind project folder."
            )

        import shlex

        project_root = self._get_wsl_project_root()
        worker_path = f"{project_root}/{self.WORKER_RELATIVE_PATH}"
        command = (
            f"source {self.VENV_PATH}/bin/activate && "
            f"exec python -u {shlex.quote(worker_path)}"
        )
        self._responses = queue.Queue()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            ["wsl.exe", "--exec", "bash", "-lc", command],
            cwd=str(self.project_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )
        response_queue = self._responses
        threading.Thread(
            target=self._read_stdout,
            args=(self._process, response_queue),
            name="speechmind-indictrans-wsl-stdout",
            daemon=True,
        ).start()
        ready = self._receive(self.startup_timeout)
        if not ready.get("ready"):
            raise RuntimeError(
                ready.get("error")
                or "IndicTrans2 WSL worker did not become ready."
            )

    def _request(self, payload: dict, timeout: float | None = None) -> dict:
        with self._lock:
            self._start()
            process = self._process
            if process is None or process.stdin is None:
                raise RuntimeError("IndicTrans2 WSL worker stdin is unavailable.")
            process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
            process.stdin.flush()
            response = self._receive(timeout or self.request_timeout)
        if not response.get("ok"):
            raise RuntimeError(
                response.get("error") or "IndicTrans2 translation request failed."
            )
        return response

    def translate_to_english(self, text: str, src_lang: str) -> dict:
        if not text or not text.strip():
            raise ValueError("Translation input is empty.")
        if not src_lang or not src_lang.strip():
            raise ValueError("Translation source language is missing.")

        cached = self._translation_cache.get(text, src_lang)
        if cached is not None:
            return {
                "translated_text": cached,
                "model": "ai4bharat/indictrans2-indic-en-dist-200M",
                "inference_seconds": 0.0,
                "cache_used": True,
                "worker_response": {"cache_hit": True},
            }

        response = self._request({
            "action": "translate",
            "text": text,
            "src_lang": src_lang,
        })
        translated = str(response.get("translated_text") or "").strip()
        if not translated:
            raise RuntimeError("IndicTrans2 returned an empty translation.")
        self._translation_cache.put(text, src_lang, translated)
        return {
            "translated_text": translated,
            "model": response.get(
                "model", "ai4bharat/indictrans2-indic-en-dist-200M"
            ),
            "inference_seconds": response.get("inference_seconds"),
            "cache_used": False,
            "worker_response": response,
        }

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
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        finally:
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
