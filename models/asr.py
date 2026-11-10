"""
SpeechMind automatic speech recognition.

Primary backend: Bodhan Indic-Transcribe when BODHAN_API_KEY is configured.
It performs automatic language handling and is designed for code-mixed speech.
Fallback: local Whisper Small, with automatic language detection. No user
language selector and no whole-recording single-language Indic routing.
"""
import torch
import whisper

from models.asr_quality import estimate_whisper_quality
from models.bodhan_client import BodhanClient


WHISPER_MODEL = "small"


class SpeechRecognizer:
    def __init__(self, backend: str = "auto", bodhan_client=None):
        if backend not in {"auto", "bodhan", "whisper"}:
            raise ValueError("ASR backend must be auto, bodhan, or whisper.")

        self.bodhan_client = bodhan_client or BodhanClient.from_env()
        self._strict_backend = backend == "bodhan"
        if backend == "bodhan" and self.bodhan_client is None:
            raise RuntimeError(
                "Bodhan backend was requested but BODHAN_API_KEY is not configured."
            )

        if backend == "auto":
            self.provider = (
                "bodhan_api" if self.bodhan_client is not None else "local_whisper"
            )
        else:
            self.provider = "bodhan_api" if backend == "bodhan" else "local_whisper"

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.whisper = None
        self.model_name = (
            "bodhan/indic-transcribe"
            if self.provider == "bodhan_api"
            else f"openai/whisper-{WHISPER_MODEL}"
        )

        if self.provider == "local_whisper":
            print(f"[ASR] Loading local Whisper-{WHISPER_MODEL} on {self.device}...")
            self.whisper = whisper.load_model(
                WHISPER_MODEL, device=str(self.device)
            )
            print("[ASR] Local Whisper loaded.")
        else:
            print(
                "[ASR] Bodhan Indic-Transcribe selected. "
                "Language selection is automatic; mixed speech is accepted."
            )

        self.last_provider_error = None

    def _decode_with_whisper(self, audio_numpy):
        return self.whisper.transcribe(
            audio_numpy,
            task="transcribe",
            language=None,
            fp16=False,
            verbose=False,
            beam_size=5,
            condition_on_previous_text=False,
        )

    @staticmethod
    def _whisper_result(whisper_result):
        transcript = str(whisper_result.get("text", "")).strip()
        detected = str(whisper_result.get("language") or "unknown").lower().strip()
        language = "en" if detected in ("en", "english") else detected
        quality = estimate_whisper_quality(
            whisper_result.get("segments", []), transcript
        )
        return {
            "transcript": transcript,
            "language": language,
            "language_profile": language,
            "language_source": "whisper_transcription_auto",
            "language_detection_score": None,
            "confidence": quality["quality_score"],
            "quality_score": quality["quality_score"],
            "quality_method": "whisper_token_quality_heuristic_uncalibrated",
            "avg_logprob": quality["avg_logprob"],
            "avg_no_speech_prob": quality["avg_no_speech_prob"],
            "avg_compression_ratio": quality["avg_compression_ratio"],
            "asr_model": "whisper-small",
            "asr_provider": "local_whisper",
            "asr_chunks": 1,
        }

    def transcribe(self, audio, language_hint=None) -> dict:
        """Transcribe naturally spoken audio; language_hint is intentionally ignored."""
        if isinstance(audio, torch.Tensor):
            audio_numpy = audio.detach().cpu().numpy()
        else:
            audio_numpy = audio
        audio_numpy = audio_numpy.squeeze()
        if audio_numpy.ndim != 1 or audio_numpy.size == 0:
            raise ValueError("ASR expects non-empty mono audio.")
        if not torch.isfinite(torch.as_tensor(audio_numpy)).all():
            raise ValueError("ASR received NaN or infinite audio.")

        if self.provider == "bodhan_api":
            try:
                result = self.bodhan_client.transcribe(audio_numpy)
                result["asr_provider"] = "bodhan_api"
                self.last_provider_error = None
                return result
            except Exception as error:
                self.last_provider_error = str(error)
                if self.bodhan_client is not None and self.whisper is None:
                    # The app's automatic mode falls back locally after a cloud
                    # outage. Explicit benchmark mode (backend='bodhan') is strict.
                    if getattr(self, "_strict_backend", False):
                        raise
                    print(f"[ASR] Bodhan request failed; using local Whisper: {error}")
                    self.provider = "local_whisper"
                    self.model_name = f"openai/whisper-{WHISPER_MODEL}"
                    self.whisper = whisper.load_model(
                        WHISPER_MODEL, device=str(self.device)
                    )

        if self.whisper is None:
            self.whisper = whisper.load_model(
                WHISPER_MODEL, device=str(self.device)
            )
        decoded = self._decode_with_whisper(audio_numpy)
        result = self._whisper_result(decoded)
        if self.last_provider_error:
            result["asr_provider_fallback_reason"] = self.last_provider_error
        return result
