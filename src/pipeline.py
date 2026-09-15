"""Hybrid de-identification pipeline: rules layer + ML layer, then redact.

Merge policy: union of both layers (recall-first — a missed PHI is the
failure mode that matters). On span conflicts the rules layer wins on type,
since it is the more precise layer for structured PHI.
"""
from __future__ import annotations

from . import rules
from .tokenize import tokenize, labels_to_spans
from .model import predict_labels


def detect(text: str, ml_bundle: dict | None) -> list[dict]:
    """Return merged entity spans [{start, end, type, text, source}]."""
    rule_ents = rules.find_entities(text)
    ml_ents: list[dict] = []
    if ml_bundle is not None:
        toks = tokenize(text)
        labels = predict_labels(ml_bundle, toks)
        for s in labels_to_spans(toks, labels, text):
            s["source"] = "ml"
            ml_ents.append(s)
    # Union: keep all rule spans; add ML spans that don't overlap a rule span.
    merged = list(rule_ents)
    for ent in ml_ents:
        if not any(ent["start"] < k["end"] and k["start"] < ent["end"] for k in merged):
            merged.append(ent)
    merged.sort(key=lambda d: (d["start"], d["end"]))
    return merged


def redact(text: str, ml_bundle: dict | None) -> tuple[str, list[dict]]:
    """Return (redacted_text, entities). Spans replaced by [ENTITY_TYPE]."""
    entities = detect(text, ml_bundle)
    out: list[str] = []
    cursor = 0
    for ent in entities:
        out.append(text[cursor:ent["start"]])
        out.append(f"[{ent['type']}]")
        cursor = ent["end"]
    out.append(text[cursor:])
    return "".join(out), entities
