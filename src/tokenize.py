"""Whitespace tokenizer with character offsets.

Keeps structured PHI (dates like 01/05/2024, phones like (555) 123-4567,
emails, MRNs) as single tokens, which suits both the regex layer (which
works on raw-text spans anyway) and the ML token classifier.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\S+")


def tokenize(text: str) -> list[dict]:
    """Return [{text, start, end}] for each whitespace-delimited token."""
    return [{"text": m.group(0), "start": m.start(), "end": m.end()}
            for m in _TOKEN_RE.finditer(text)]


def label_tokens(tokens: list[dict], entities: list[dict]) -> list[str]:
    """Assign each token an entity type (or 'O') by max span overlap."""
    labels = []
    for tok in tokens:
        best, best_overlap = "O", 0
        for ent in entities:
            overlap = min(tok["end"], ent["end"]) - max(tok["start"], ent["start"])
            if overlap > best_overlap:
                best, best_overlap = ent["type"], overlap
        labels.append(best)
    return labels


def labels_to_spans(tokens: list[dict], labels: list[str], text: str) -> list[dict]:
    """Merge consecutive same-type token labels into entity spans.

    Adjacent tokens (separated only by whitespace) with the same label are
    merged; the span text is sliced from the original string so offsets stay
    exact.
    """
    groups: list[list[int]] = []
    for i, lab in enumerate(labels):
        if lab == "O":
            continue
        if groups and labels[groups[-1][-1]] == lab \
                and text[tokens[groups[-1][-1]]["end"]:tokens[i]["start"]].strip() == "":
            groups[-1].append(i)
        else:
            groups.append([i])
    spans = []
    for g in groups:
        start = tokens[g[0]]["start"]
        end = tokens[g[-1]]["end"]
        # Strip trailing punctuation glued to the last token ("Hayes." -> "Hayes")
        # so ML spans line up with the gold annotations. Offsets stay exact.
        while end > start and text[end - 1] in ".,:;!?":
            end -= 1
        spans.append({"start": start, "end": end,
                      "type": labels[g[0]], "text": text[start:end]})
    return spans
