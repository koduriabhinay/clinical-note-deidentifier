"""ML token classifier: training, model selection, and entity-level evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

from .tokenize import tokenize, label_tokens
from .features import featurize_tokens

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced",
                                              n_jobs=-1),
    "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            n_jobs=-1, random_state=42),
}


def build_matrix(notes: list[dict]):
    """Featurize all tokens of all notes. Returns (X_dicts, y, groups)."""
    dicts, y, groups = [], [], []
    for gi, note in enumerate(notes):
        toks = tokenize(note["text"])
        labs = label_tokens(toks, note["entities"])
        dicts.extend(featurize_tokens(toks))
        y.extend(labs)
        groups.extend([gi] * len(toks))
    return dicts, np.array(y), np.array(groups)


def compare_models(X_dicts, y, groups, cv: int = 3) -> dict:
    """3-fold GroupKFold (grouped by note) token-level micro-F1 comparison."""
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dicts)
    gkf = GroupKFold(n_splits=cv)
    scores = {}
    for name, clf in CANDIDATES.items():
        s = cross_val_score(clf, X, y, groups=groups, cv=gkf,
                            scoring="f1_micro", n_jobs=1)
        scores[name] = {"mean": float(s.mean()), "std": float(s.std()),
                        "folds": [float(v) for v in s]}
    return scores


def train_final(X_dicts, y, model_name: str):
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dicts)
    clf = CANDIDATES[model_name]
    clf.fit(X, y)
    return {"vectorizer": vec, "classifier": clf, "model_name": model_name,
            "classes": list(clf.classes_)}


def predict_labels(bundle: dict, tokens: list[dict]) -> list[str]:
    from .features import featurize_tokens as _ft
    X = bundle["vectorizer"].transform(_ft(tokens))
    return list(bundle["classifier"].predict(X))


# ---------------------------------------------------------------------------
# Entity-level evaluation (exact span + type match)
# ---------------------------------------------------------------------------
def _key(e: dict) -> tuple:
    return (e["start"], e["end"], e["type"])


def evaluate_entities(gold_notes: list[dict], pred_per_note: list[list[dict]]) -> dict:
    per_type: dict[str, dict[str, int]] = {}
    for note, preds in zip(gold_notes, pred_per_note):
        gold_keys = {_key(e) for e in note["entities"]}
        pred_keys = {_key(e) for e in preds}
        types = {e["type"] for e in note["entities"]} | {e["type"] for e in preds}
        for t in types:
            d = per_type.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
            g = {k for k in gold_keys if k[2] == t}
            p = {k for k in pred_keys if k[2] == t}
            d["tp"] += len(g & p)
            d["fp"] += len(p - g)
            d["fn"] += len(g - p)
    report: dict[str, dict] = {}
    tot_tp = tot_fp = tot_fn = 0
    for t in sorted(per_type):
        d = per_type[t]
        p = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0.0
        r = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        report[t] = {"precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(f1, 4), "support": d["tp"] + d["fn"]}
        tot_tp += d["tp"]
        tot_fp += d["fp"]
        tot_fn += d["fn"]
    mp = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else 0.0
    mr = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn else 0.0
    mf1 = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    report["_micro"] = {"precision": round(mp, 4), "recall": round(mr, 4),
                        "f1": round(mf1, 4)}
    return report
