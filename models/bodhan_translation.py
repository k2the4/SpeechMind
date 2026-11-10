"""Optional Bodhan Indic-Translate API adapter for mixed-language transcripts.

This is opt-in because transcript text is sent to a third-party service and
usage is metered. The prompt deliberately asks for meaning-preserving English,
not transliteration or a summary.
"""
from __future__ import annotations

import os

import requests

API_URL = "https://api.bodhan.ai/v1/chat/completions"
MODEL_NAME = "indic-translate"


class BodhanTranslator:
    def __init__(self, api_key: str | None = None, timeout: int = 120):
        self.api_key = api_key or os.getenv("BODHAN_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "SPEECHMIND_TRANSLATOR_PROVIDER=bodhan requires BODHAN_API_KEY. "
                "Transcript text is sent to Bodhan AI and usage is metered."
            )
        self.timeout = timeout

    def to_english(self, transcript: str, src_lang: str | None = None) -> str:
        text = str(transcript or "").strip()
        if not text:
            return text

        prompt = (
            "Translate the following spoken transcript into natural English. "
            "The input may mix Hindi, English, Romanized Hindi, and other Indian "
            "language words within the same sentence. Translate the meaning, not "
            "the script. Preserve names, numbers, tense, negation, intent, and "
            "important nuance. Do not summarize, explain, or add information. "
            "If the transcript is already English, return it unchanged.\n\n"
            f"Transcript:\n{text}"
        )
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "target_language_code": "en",
            },
            timeout=self.timeout,
        )
        if response.status_code == 429:
            raise RuntimeError(
                "Bodhan translation API rate limit or credit limit reached (HTTP 429)."
            )
        response.raise_for_status()
        payload = response.json()
        try:
            translated = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "Bodhan translation API returned an unexpected response."
            ) from error
        translated = str(translated).strip()
        if not translated:
            raise RuntimeError("Bodhan translation API returned an empty translation.")
        return translated
