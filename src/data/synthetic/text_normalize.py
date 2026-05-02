"""SmartHandover - Synthetic-text post-processing helpers.

Centralises the small string-cleanup steps so every consumer of the
synthetic JSONL gets the same canonical form. Currently:

  * Map "smart" punctuation (curly quotes, em-dashes, ellipsis,
    non-breaking spaces) to the closest ASCII equivalent.
  * Collapse runs of whitespace.
  * Strip leading/trailing whitespace.

Why
---
GPT-4o tends to emit U+2018/U+2019/U+201C/U+201D for quotes, U+2013/U+2014
for dashes, and U+2026 for ellipsis. The MELD ground-truth used to train
the RoBERTa text classifier is plain ASCII, so leaving smart punctuation
in the synthetic data would cause:

  - Tokenizer mismatch (BPE encodes ' and U+2019 as different tokens).
  - Confusion-prone display in Windows cp1252 consoles.
  - VADER lexicon misses on contractions like ``don't`` -> ``don’t``.

Importing this module is cheap (no third-party deps).
"""

from __future__ import annotations

import re
from typing import Dict


# Mapping is intentionally minimal: only characters we have actually seen
# in GPT-4o output for English text. Add entries here if new ones surface.
_PUNCT_FIXES: Dict[str, str] = {
    "‘": "'",   # left single quote
    "’": "'",   # right single quote / apostrophe
    "‚": "'",   # single low-9 quote
    "‛": "'",   # single high-reversed-9 quote
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "„": '"',   # double low-9 quote
    "–": "-",   # en dash
    "—": "-",   # em dash
    "―": "-",   # horizontal bar
    "…": "...", # horizontal ellipsis
    " ": " ",   # non-breaking space
    " ": " ",   # thin space
    "​": "",    # zero-width space (just drop)
    "‌": "",    # zero-width non-joiner
    "‍": "",    # zero-width joiner
    "﻿": "",    # BOM if a stray one slipped through
}

_WS_RE = re.compile(r"[ \t]+")


def normalise(text: str) -> str:
    """Return ``text`` with smart punctuation flattened to ASCII.

    Idempotent: ``normalise(normalise(t)) == normalise(t)``.
    """
    if not text:
        return ""
    out = text
    for src, dst in _PUNCT_FIXES.items():
        if src in out:
            out = out.replace(src, dst)
    # Collapse only horizontal whitespace; keep newlines intact
    # (the generator output is single-line so this is mostly belt-and-braces).
    out = _WS_RE.sub(" ", out)
    return out.strip()


def normalise_record(rec: dict, field: str = "text") -> dict:
    """Return a shallow copy of ``rec`` with ``rec[field]`` normalised."""
    if field not in rec:
        return rec
    new = dict(rec)
    new[field] = normalise(str(rec[field]))
    return new
