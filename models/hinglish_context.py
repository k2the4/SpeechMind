"""Conservative vocabulary extraction for automatic Hinglish ASR context.

Whisper is used only as a short-window vocabulary scout. Its transcript is
never used as the final transcript and only exact matches from this small
technical vocabulary are passed to Srota.
"""
from __future__ import annotations

import re
import time
import unicodedata


# Canonical terms and common spoken/written variants. Keep this list small:
# adding a term means it may bias decoding when Whisper hears it in the audio.
_CONTEXT_ALIASES = {
    "layout": ("layout", "lay-out"),
    "interface": ("interface",),
    "wireframe": ("wireframe", "wire frame"),
    "prototype": ("prototype",),
    "figma": ("figma",),
    "design": ("design",),
    "project": ("project",),
    "website": ("website", "web site"),
    "app": ("app",),
    "screen": ("screen",),
    "dashboard": ("dashboard",),
    "frontend": ("frontend", "front end"),
    "backend": ("backend", "back end"),
    "software": ("software",),
    "code": ("code",),
    "python": ("python",),
    "javascript": ("javascript", "java script"),
    "api": ("api",),
    "database": ("database", "data base"),
    "model": ("model",),
    "dataset": ("dataset", "data set"),
    "algorithm": ("algorithm",),
    "presentation": ("presentation",),
    "report": ("report",),
    "excel": ("excel",),
    "powerpoint": ("powerpoint", "power point"),
    "audio": ("audio",),
    "video": ("video",),
    "file": ("file",),
}

_CANONICAL = {
    "javascript": "JavaScript",
    "api": "API",
    "figma": "Figma",
    "python": "Python",
    "excel": "Excel",
    "powerpoint": "PowerPoint",
}
_CANONICAL.update({term: term for term in _CONTEXT_ALIASES if term not in _CANONICAL})


def extract_hinglish_context_terms(
    transcript: str,
    *,
    max_terms: int = 6,
) -> list[str]:
    """Return only exact, allow-listed English terms heard in a scout transcript."""
    if not transcript or max_terms <= 0:
        return []

    text = unicodedata.normalize("NFKC", str(transcript)).casefold()
    matches: list[tuple[int, str]] = []

    for canonical, aliases in _CONTEXT_ALIASES.items():
        positions = []
        for alias in aliases:
            pattern = r"(?<![\w])" + re.escape(alias.casefold()) + r"(?![\w])"
            match = re.search(pattern, text, flags=re.UNICODE)
            if match:
                positions.append(match.start())
        if positions:
            matches.append((min(positions), _CANONICAL[canonical]))

    matches.sort(key=lambda item: item[0])
    # De-duplicate while preserving the order heard in the scout transcript.
    result: list[str] = []
    for _, term in matches:
        if term not in result:
            result.append(term)
        if len(result) >= max_terms:
            break
    return result


def build_hinglish_context(transcript: str, *, max_terms: int = 6) -> str:
    """Return a compact comma-separated Qwen/Srota context string."""
    return ", ".join(
        extract_hinglish_context_terms(transcript, max_terms=max_terms)
    )


def infer_hinglish_context_from_audio_tail(
    audio,
    whisper_model,
    *,
    sample_rate: int = 16000,
    tail_seconds: float = 3.0,
) -> dict:
    """Auto-detect the final speech segment and extract terms only if it is English."""
    if sample_rate <= 0 or tail_seconds <= 0:
        return {
            "context": "",
            "source": "whisper_tail_too_short",
            "probe_seconds": 0.0,
            "transcript": "",
        }
    sample_count = int(getattr(audio, "size", len(audio)))
    tail_samples = min(sample_count, int(sample_rate * tail_seconds))
    if tail_samples < sample_rate:
        return {
            "context": "",
            "source": "whisper_tail_too_short",
            "probe_seconds": 0.0,
            "transcript": "",
        }

    started = time.perf_counter()
    try:
        tail_audio = audio[-tail_samples:]
        result = whisper_model.transcribe(
            tail_audio,
            task="transcribe",
            language=None,
            fp16=False,
            verbose=False,
            beam_size=1,
            condition_on_previous_text=False,
        )
        transcript = str(result.get("text") or "").strip()
        detected_language = str(result.get("language") or "").lower().strip()
        if detected_language not in {"en", "english"}:
            return {
                "context": "",
                "source": "whisper_tail_not_english",
                "probe_seconds": round(time.perf_counter() - started, 3),
                "transcript": transcript,
            }
        context = build_hinglish_context(transcript)
        return {
            "context": context,
            "source": (
                "whisper_tail_vocabulary"
                if context else "whisper_tail_no_allowlisted_terms"
            ),
            "probe_seconds": round(time.perf_counter() - started, 3),
            "transcript": transcript,
        }
    except Exception as error:
        return {
            "context": "",
            "source": f"whisper_tail_probe_failed:{type(error).__name__}",
            "probe_seconds": round(time.perf_counter() - started, 3),
            "transcript": "",
        }
