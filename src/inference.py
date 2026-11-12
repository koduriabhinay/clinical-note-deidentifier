"""Single-note inference: load trained artifacts, redact text."""
from __future__ import annotations

import joblib

from . import pipeline

_ARTIFACT_PATH = "models/artifacts.joblib"


def load_artifacts(path: str = _ARTIFACT_PATH) -> dict:
    return joblib.load(path)


def redact_note(text: str, artifacts: dict) -> tuple[str, list[dict]]:
    return pipeline.redact(text, artifacts["ml_bundle"])
