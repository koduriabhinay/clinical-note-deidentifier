"""End-to-end test suite for the clinical-note-deidentifier pipeline."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import rules
from src.data_gen import generate_dataset
from src.features import featurize_tokens, token_features, word_shape
from src.model import evaluate_entities
from src.pipeline import detect, redact
from src.tokenize import label_tokens, labels_to_spans, tokenize


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def test_dataset_size_and_split():
    ds = generate_dataset()
    assert len(ds["train"]) == 960
    assert len(ds["test"]) == 240
    assert ds["meta"]["synthetic"] is True


def test_dataset_deterministic():
    a = generate_dataset(seed=123)
    b = generate_dataset(seed=123)
    assert a["train"][0]["text"] == b["train"][0]["text"]
    assert a["test"][5]["entities"] == b["test"][5]["entities"]


def test_entity_offsets_match_text():
    ds = generate_dataset()
    for note in ds["train"][:20] + ds["test"][:20]:
        for e in note["entities"]:
            assert note["text"][e["start"]:e["end"]] == e["text"]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def test_tokenize_offsets_are_exact():
    text = "Patient: John Smith DOB 01/05/1980."
    toks = tokenize(text)
    for t in toks:
        assert text[t["start"]:t["end"]] == t["text"]
    # full coverage: joining tokens with inter-token gaps reconstructs text
    rebuilt = toks[0]["text"]
    for prev, cur in zip(toks, toks[1:]):
        rebuilt += text[prev["end"]:cur["start"]] + cur["text"]
    assert rebuilt == text


def test_labels_to_spans_strips_trailing_punctuation():
    text = "Seen by Dr. Alan Price. Vitals stable."
    toks = tokenize(text)
    labels = ["O", "O", "PROVIDER", "PROVIDER", "PROVIDER", "O", "O"]
    spans = labels_to_spans(toks, labels, text)
    assert len(spans) == 1
    assert spans[0]["text"] == "Dr. Alan Price"
    assert text[spans[0]["start"]:spans[0]["end"]] == "Dr. Alan Price"


def test_no_glued_tokens_in_generated_notes():
    # Regression test: template 10 used to emit "SullivanPatient" (missing
    # separator between the provider name and the following sentence).
    import re
    ds = generate_dataset()
    bad = [n["text"][:80] for n in ds["train"] + ds["test"]
           if re.search(r"[A-Za-z]Patient\b", n["text"])]
    assert bad == [], f"glued '...Patient' found in {len(bad)} notes: {bad[:3]}"


def test_labels_to_spans_merges_adjacent_names():
    text = "Patient: Mary Johnson here"
    toks = tokenize(text)
    labels = ["O", "PATIENT", "PATIENT", "O"]
    spans = labels_to_spans(toks, labels, text)
    assert len(spans) == 1
    assert spans[0]["text"] == "Mary Johnson"
    assert spans[0]["type"] == "PATIENT"


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def test_word_shape_collapses_runs():
    assert word_shape("Smith") == "Xx"  # "Xxxx" with runs collapsed
    assert word_shape("123-4567") == "d-d"  # runs collapse: "ddd-dddd" -> "d-d"


def test_token_features_orthographic_flags():
    toks = tokenize("Call (555) 123-4567 today")
    f = token_features(toks, 1)
    assert f["has_digit"] is True
    assert f["is_title"] is False


def test_featurize_tokens_count_matches():
    toks = tokenize("Dr. Jane Doe, MRN 12345678")
    feats = featurize_tokens(toks)
    assert len(feats) == len(toks)
    assert all("bias" in d and "shape" in d for d in feats)


# ---------------------------------------------------------------------------
# Rule layer
# ---------------------------------------------------------------------------
def test_rules_detect_structured_phi():
    text = "Call (415) 555-1234 or jane.doe@example.com. MRN: 12345678. DOB 01/05/1980."
    ents = rules.find_entities(text)
    types = {e["type"] for e in ents}
    assert {"PHONE", "EMAIL", "MRN", "DATE"} <= types
    for e in ents:
        assert text[e["start"]:e["end"]] == e["text"]
        assert e["source"] == "rules"


def test_rules_do_not_match_names():
    ents = rules.find_entities("Patient Mary Johnson seen by Dr. Alan Price.")
    assert ents == []


# ---------------------------------------------------------------------------
# Hybrid pipeline (needs trained artifacts)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def artifacts():
    path = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts.joblib")
    if not os.path.exists(path):
        pytest.skip("trained artifacts not present; run scripts/train.py first")
    import joblib
    return joblib.load(path)


def test_hybrid_detects_both_phi_kinds(artifacts):
    bundle = artifacts["ml_bundle"]
    text = "Patient Mary Johnson, DOB 01/05/1980, MRN 12345678."
    ents = detect(text, bundle)
    types = {e["type"] for e in ents}
    assert "DATE" in types and "MRN" in types
    name_hits = [e for e in ents if e["type"] in ("PATIENT", "PROVIDER")]
    assert len(name_hits) >= 1


def test_redact_replaces_with_type_tokens(artifacts):
    bundle = artifacts["ml_bundle"]
    text = "Patient Mary Johnson, DOB 01/05/1980."
    redacted, ents = redact(text, bundle)
    assert "Mary Johnson" not in redacted
    assert "01/05/1980" not in redacted
    assert "[DATE]" in redacted
    assert len(ents) >= 2


# ---------------------------------------------------------------------------
# Evaluation math
# ---------------------------------------------------------------------------
def test_evaluate_entities_perfect_score():
    note = {"text": "Call 555-1234", "entities": [
        {"start": 5, "end": 13, "type": "PHONE", "text": "555-1234"}]}
    report = evaluate_entities([note], [[{"start": 5, "end": 13, "type": "PHONE",
                                          "text": "555-1234"}]])
    assert report["PHONE"]["f1"] == 1.0
    assert report["_micro"]["f1"] == 1.0


def test_evaluate_entities_miss_penalized():
    note = {"text": "Call 555-1234", "entities": [
        {"start": 5, "end": 13, "type": "PHONE", "text": "555-1234"}]}
    report = evaluate_entities([note], [[]])
    assert report["PHONE"]["recall"] == 0.0
    assert report["PHONE"]["support"] == 1


# ---------------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------------
def test_api_redact_endpoint(artifacts):
    os.environ["ARTIFACTS_PATH"] = os.path.join(
        os.path.dirname(__file__), "..", "models", "artifacts.joblib")
    from fastapi.testclient import TestClient
    import app as service
    client = TestClient(service.app)
    r = client.get("/health")
    assert r.status_code == 200
    r = client.post("/redact", json={"text": "Patient Mary Johnson, DOB 01/05/1980."})
    assert r.status_code == 200
    body = r.json()
    assert body["n_entities"] >= 2
    assert "Mary Johnson" not in body["redacted_text"]
    assert any(e["type"] == "DATE" for e in body["entities"])
