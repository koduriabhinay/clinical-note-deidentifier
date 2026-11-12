"""Rule-based PHI detection layer (high precision, raw-text regexes).

Covers the structured PHI types: PHONE, DATE, MRN, EMAIL, ADDRESS.
Names (PATIENT, PROVIDER) are left to the ML layer — they have no reliable
surface pattern.

Rules run first; on span conflicts the rule layer wins (it is the more
precise of the two on these types).
"""
from __future__ import annotations

import re

MONTHS = ("January|February|March|April|May|June|July|August|September|"
          "October|November|December")

PATTERNS: list[tuple[str, str]] = [
    # MRN must be checked before PHONE so "MRN: 12345678" is not half-matched.
    ("MRN", r"\bMRN\s*[:#]?\s*(\d{6,8})\b"),
    ("PHONE", r"(?<!\d)(?:\+1\s*)?(?:\(\d{3}\)\s*|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?!\d)"),
    ("DATE", rf"\b\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}\b"),
    ("DATE", rf"\b(?:{MONTHS})\s+\d{{1,2}},?\s+\d{{4}}\b"),
    ("DATE", r"\b\d{4}-\d{2}-\d{2}\b"),
    ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("ADDRESS",
     r"\b\d{1,5}\s+[A-Z][A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|"
     r"Boulevard|Blvd|Lane|Ln|Drive|Dr),\s+[A-Z][A-Za-z]+,\s+[A-Z]{2}\s+\d{5}\b"),
]

_COMPILED = [(typ, re.compile(pat)) for typ, pat in PATTERNS]


def find_entities(text: str) -> list[dict]:
    """Return [{start, end, type, text, source}] sorted by start offset."""
    found: list[dict] = []
    for typ, rx in _COMPILED:
        for m in rx.finditer(text):
            # For MRN, capture group 1 is the number itself.
            s, e = (m.start(1), m.end(1)) if typ == "MRN" and m.lastindex else (m.start(), m.end())
            found.append({"start": s, "end": e, "type": typ,
                          "text": text[s:e], "source": "rules"})
    # Dedupe overlaps: earliest start wins, then longest span.
    found.sort(key=lambda d: (d["start"], -(d["end"] - d["start"])))
    kept: list[dict] = []
    for ent in found:
        if not any(ent["start"] < k["end"] and k["start"] < ent["end"] for k in kept):
            kept.append(ent)
    return kept
