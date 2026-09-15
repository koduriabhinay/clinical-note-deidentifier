"""End-to-end training: generate data -> train ML layer -> evaluate hybrid."""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib

from src.data_gen import generate_dataset, SEED, N_NOTES
from src.model import build_matrix, compare_models, train_final, evaluate_entities
from src.pipeline import detect

MODEL_NAME = "logistic_regression"  # winner of scripts/compare_models.py


def main() -> None:
    t0 = time.time()
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("generating synthetic dataset ...")
    dataset = generate_dataset()
    with open("data/dataset.json", "w") as f:
        json.dump(dataset, f)
    train_notes, test_notes = dataset["train"], dataset["test"]
    print(f"  train={len(train_notes)} test={len(test_notes)} (seed {SEED})")

    print("featurizing ...")
    X_dicts, y, groups = build_matrix(train_notes)
    print(f"  {len(y)} tokens, {len(set(y))} classes: {sorted(set(y))}")

    print("comparing candidate models (3-fold GroupKFold, token micro-F1) ...")
    comp = compare_models(X_dicts, y, groups)
    for name, s in comp.items():
        print(f"  {name}: {s['mean']:.4f} ± {s['std']:.4f}")
    winner = max(comp, key=lambda n: comp[n]["mean"])
    print(f"  winner: {winner}")

    print("training final model ...")
    bundle = train_final(X_dicts, y, winner)
    joblib.dump({"ml_bundle": bundle}, "models/artifacts.joblib")

    print("evaluating hybrid pipeline on held-out test notes ...")
    preds = [detect(note["text"], bundle) for note in test_notes]
    report = evaluate_entities(test_notes, preds)
    for t, m in report.items():
        print(f"  {t}: P={m['precision']} R={m['recall']} F1={m['f1']} "
              f"(support {m.get('support', '-')})")

    metrics = {
        "dataset": {"n_notes": N_NOTES, "seed": SEED,
                    "train_notes": len(train_notes), "test_notes": len(test_notes),
                    "synthetic": True},
        "model_selection": comp,
        "chosen_model": winner,
        "entity_metrics": report,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"done in {time.time() - t0:.1f}s -> models/artifacts.joblib, metrics.json")


if __name__ == "__main__":
    main()
