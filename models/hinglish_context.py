"""Conservative vocabulary extraction for automatic Hinglish ASR context.

Whisper is used only as a short-window vocabulary scout. Its transcript is
never used as the final transcript and only exact matches from this small
technical vocabulary are passed to Srota.
"""
from __future__ import annotations

import re
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
