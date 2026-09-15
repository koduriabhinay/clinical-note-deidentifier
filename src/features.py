"""Token-level features for the ML PHI classifier.

Orthographic + context + gazetteer features, vectorized with sklearn's
DictVectorizer into a sparse matrix. Offline-friendly: no embeddings,
no downloads.
"""
from __future__ import annotations

from .data_gen import GAZETTEER_FIRST, GAZETTEER_LAST

_FIRST = {n.lower() for n in GAZETTEER_FIRST}
_LAST = {n.lower() for n in GAZETTEER_LAST}
_TITLES = {"dr", "doctor", "mr", "mrs", "ms"}
_CREDENTIALS = {"md", "do", "rn", "pa", "np", "phd"}
_PATIENT_CUES = {"patient", "pt", "name"}
_DOB_CUES = {"dob", "birth"}


def word_shape(tok: str) -> str:
    out = []
    for ch in tok:
        if ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        elif ch.isdigit():
            out.append("d")
        else:
            out.append(ch)
    shape = "".join(out)
    # collapse runs: "Xxxx", "ddd"
    collapsed = []
    for c in shape:
        if not collapsed or collapsed[-1] != c or c not in "Xxd":
            collapsed.append(c)
    return "".join(collapsed)


def _clean(tok: str) -> str:
    return tok.strip(".,:;()").lower()


def token_features(tokens: list[dict], i: int) -> dict:
    tok = tokens[i]["text"]
    prev_t = tokens[i - 1]["text"] if i > 0 else "<START>"
    next_t = tokens[i + 1]["text"] if i < len(tokens) - 1 else "<END>"
    c, pc, nc = _clean(tok), _clean(prev_t), _clean(next_t)
    return {
        "bias": 1.0,
        "lower": c,
        "is_title": tok.istitle(),
        "is_upper": tok.isupper(),
        "is_digit": tok.isdigit(),
        "has_digit": any(ch.isdigit() for ch in tok),
        "len": len(tok),
        "prefix3": c[:3],
        "suffix3": c[-3:],
        "shape": word_shape(tok),
        "prev_lower": pc,
        "prev_shape": word_shape(prev_t),
        "next_lower": nc,
        "next_shape": word_shape(next_t),
        "in_first_gaz": c in _FIRST,
        "in_last_gaz": c in _LAST,
        "prev_is_title": pc in _TITLES,
        "next_is_credential": nc in _CREDENTIALS,
        "prev_is_patient_cue": pc in _PATIENT_CUES,
        "prev_is_dob_cue": pc in _DOB_CUES,
        "next_is_title": nc in _TITLES,
    }


def featurize_tokens(tokens: list[dict]) -> list[dict]:
    return [token_features(tokens, i) for i in range(len(tokens))]
