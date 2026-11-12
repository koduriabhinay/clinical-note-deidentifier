"""Pretty-print the saved metrics from scripts/train.py."""
import json

with open("metrics.json") as f:
    m = json.load(f)

print(f"Dataset: {m['dataset']['n_notes']} synthetic notes "
      f"(train={m['dataset']['train_notes']}, test={m['dataset']['test_notes']})")
print(f"Chosen model: {m['chosen_model']}")
print("\nModel selection (3-fold GroupKFold token micro-F1):")
for name, s in m["model_selection"].items():
    print(f"  {name:22s} {s['mean']:.4f} ± {s['std']:.4f}")
print("\nHeld-out entity-level metrics (exact span + type match):")
print(f"  {'type':10s} {'P':>7s} {'R':>7s} {'F1':>7s} {'support':>7s}")
for t, r in m["entity_metrics"].items():
    sup = r.get("support", "-")
    print(f"  {t:10s} {r['precision']:7.4f} {r['recall']:7.4f} "
          f"{r['f1']:7.4f} {sup:>7}")
