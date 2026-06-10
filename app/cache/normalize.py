"""Query normalization and stable hashes for cache keys."""

from __future__ import annotations

import hashlib
import re


def normalize_query(text: str) -> str:
    """Lowercase, strip, collapse whitespace (orchestrator smalltalk style)."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def short_hash(*parts: str) -> str:
    """16-char hex prefix of sha256 over joined parts."""
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
