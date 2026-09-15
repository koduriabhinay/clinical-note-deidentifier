"""FastAPI service: PHI de-identification endpoint."""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import load_artifacts, redact_note

app = FastAPI(title="clinical-note-deidentifier",
              description="Hybrid regex + ML PHI de-identification for clinical notes.")

_artifacts = None


def get_artifacts():
    global _artifacts
    if _artifacts is None:
        path = os.environ.get("ARTIFACTS_PATH", "models/artifacts.joblib")
        _artifacts = load_artifacts(path)
    return _artifacts


class RedactRequest(BaseModel):
    text: str


class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int
    source: str


class RedactResponse(BaseModel):
    redacted_text: str
    entities: list[Entity]
    n_entities: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/redact", response_model=RedactResponse)
def redact(req: RedactRequest):
    redacted, entities = redact_note(req.text, get_artifacts())
    return RedactResponse(
        redacted_text=redacted,
        entities=[Entity(text=e["text"], type=e["type"], start=e["start"],
                         end=e["end"], source=e["source"]) for e in entities],
        n_entities=len(entities),
    )
