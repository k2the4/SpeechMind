"""Small bounded cache for completed translation results.

This is a result cache only. It does not enable decoder KV/past-key-value
caching, which is disabled for IndicTrans2 due to its custom decoder bug.
"""
from __future__ import annotations

from collections import OrderedDict
from threading import RLock


class TranslationResultCache:
    def __init__(self, max_items: int = 64):
        if max_items < 1:
            raise ValueError("max_items must be positive.")
        self.max_items = int(max_items)
        self._items = OrderedDict()
        self._lock = RLock()

    def get(self, text: str, source_language: str):
        key = (str(source_language or "").strip(), str(text or "").strip())
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, text: str, source_language: str, translated_text: str) -> None:
        key = (str(source_language or "").strip(), str(text or "").strip())
        value = str(translated_text or "").strip()
        if not all(key) or not value:
            return
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self):
        with self._lock:
            return len(self._items)
