# clinical-note-deidentifier

Hybrid PHI (protected health information) de-identification pipeline for clinical notes:
a **regex/rules layer** for structured PHI plus a **trained scikit-learn token classifier**
for person names, merged recall-first and served behind a FastAPI endpoint.

> **Synthetic by design.** Every name, date, phone number, MRN, address, and email in
> this project is fictitious — generated programmatically from fixed pools with a fixed
> random seed (`seed=42`). **No real patient data is used, stored, or produced anywhere
> in this repository.**

## Results (held-out test set, exact span + type match)

| Entity type | Precision | Recall | F1 | Support |
|-------------|-----------|--------|----|---------|
| ADDRESS | 1.0000 | 1.0000 | 1.0000 | 24 |
| DATE | 1.0000 | 1.0000 | 1.0000 | 226 |
| EMAIL | 1.0000 | 1.0000 | 1.0000 | 24 |
| MRN | 1.0000 | 1.0000 | 1.0000 | 53 |
| PATIENT | 1.0000 | 1.0000 | 1.0000 | 240 |
| PHONE | 1.0000 | 1.0000 | 1.0000 | 85 |
| PROVIDER | 1.0000 | 1.0000 | 1.0000 | 155 |
| **micro avg** | **1.0000** | **1.0000** | **1.0000** | 807 |

Model selection (3-fold GroupKFold grouped by note, token-level micro-F1):

| Model | Mean F1 | Std |
|-------|---------|-----|
| logistic_regression | 0.9999 | 0.0001 |
| **random_forest** (selected) | 1.0000 | 0.0000 |

- **17/17 pytest tests pass** (`pytest tests/`)
- Held-out names are genuinely unseen: test notes use name pools disjoint from training.
- Honest caveat: perfect scores reflect the regularity of the *synthetic* note templates.
  Real clinical text is messier — abbreviations, typos, and dictated notes would degrade
  these numbers. The value here is the pipeline architecture and the eval harness, not
  the absolute scores.

## Architecture

```
clinical note text
      │
      ├─► rules layer (src/rules.py) ── regexes for PHONE, DATE, MRN, EMAIL, ADDRESS
      │                                  (high precision, raw-text spans)
      │
      └─► ML layer ── whitespace tokenizer w/ char offsets (src/tokenize.py)
                       orthographic + context + gazetteer features (src/features.py)
                       DictVectorizer → RandomForest (src/model.py)
                       (handles PATIENT / PROVIDER names — no reliable surface pattern)
              │
              ▼
   merge: union of both layers (recall-first — a missed PHI is the failure
   that matters); on span conflicts the rules layer wins on type.
              │
              ▼
   redact: spans replaced with entity-type-aware tokens, e.g. [PATIENT], [DATE]
```

Example:

```
Input:    Patient Mary Johnson, DOB 01/05/1980. Call (415) 555-1234.
Output:   Patient [PATIENT], DOB [DATE]. Call [PHONE].
```

## Project layout

```
src/
  data_gen.py    synthetic note generator (seed=42, 1200 notes, char-level gold spans)
  tokenize.py    whitespace tokenizer with exact character offsets + span merging
  features.py    orthographic / context / gazetteer token features (offline, no downloads)
  rules.py       regex layer for structured PHI
  model.py       training, model selection, entity-level eval harness
  pipeline.py    hybrid detect() + redact()
  inference.py   load artifacts, redact one note
scripts/
  train.py       end-to-end: generate → featurize → compare models → train → evaluate
  evaluate.py    pretty-print metrics.json
  redact.py      redact one note from the CLI
tests/
  test_pipeline.py   17 tests: data gen, tokenizer, features, rules, hybrid,
                     eval math, FastAPI endpoint
app.py           FastAPI service: POST /redact, GET /health
Dockerfile       trains the pipeline at build time (fully offline, deterministic)
```

## Run it

```bash
pip install -r requirements.txt

# Train + evaluate (regenerates the synthetic data, ~80s)
python scripts/train.py
python scripts/evaluate.py

# Run the tests
pytest tests/ -q

# Redact a note from the CLI
python scripts/redact.py "Patient Mary Johnson, DOB 01/05/1980."

# Serve the API
uvicorn app:app --port 8000
curl -X POST localhost:8000/redact -H 'Content-Type: application/json' \
  -d '{"text": "Patient Mary Johnson, DOB 01/05/1980."}'
```

Or with Docker (trains inside the image build — no data or model files needed):

```bash
docker build -t clinical-note-deidentifier .
docker run -p 8000:8000 clinical-note-deidentifier
```

## Tech

Python, scikit-learn, regex/NLP, FastAPI, Pydantic, Docker, pytest. No deep-learning
frameworks, no model downloads — the whole pipeline trains and runs offline.
