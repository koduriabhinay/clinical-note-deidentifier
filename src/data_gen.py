"""Synthetic clinical note generator (deterministic).

Generates clinical notes with programmatically injected PHI entities and
character-level annotations.

SYNTHETIC BY DESIGN: every name, date, phone number, MRN, address, and email
in the generated data is fictitious, drawn from fixed pools with a fixed
random seed. No real patient data is used or produced.
"""
from __future__ import annotations

import json
import random

SEED = 42
N_NOTES = 1200
TRAIN_FRAC = 0.8

# ---------------------------------------------------------------------------
# Name pools. The first 50 entries of each list form the gazetteer used as an
# ML feature; entries 0-99 are used for TRAINING notes and entries 100-149
# for TEST notes, so the held-out evaluation measures generalization to
# unseen names.
# ---------------------------------------------------------------------------
FIRST_NAMES = (
    "James Mary John Patricia Robert Jennifer Michael Linda William Elizabeth "
    "David Barbara Richard Susan Joseph Jessica Thomas Sarah Charles Karen "
    "Christopher Nancy Daniel Lisa Matthew Betty Anthony Margaret Mark Sandra "
    "Donald Ashley Steven Kimberly Paul Emily Andrew Donna Joshua Michelle "
    "Kenneth Carol Kevin Amanda Brian Melissa George Deborah Ronald Stephanie "
    "Timothy Rebecca Jason Sharon Jeffrey Laura Ryan Cynthia Jacob Kathryn "
    "Gary Amy Nicholas Shirley Eric Angela Jonathan Brenda Stephen Pamela "
    "Larry Emma Justin Nicole Scott Anna Brandon Heather Bentley Audrey "
    "Samuel Kelly Frank Dylan Gregory Rachel Raymond Carolyn Alexander Maria "
    "Patrick Gloria Jack Jacqueline Dennis Hannah Jerry Martha Tyler Martha "
    "Aaron Lauren Jose Evelyn Nathan Joan Henry Judith Douglas Julie Peter "
    "Kelly Ethan Ruth Walter Virginia Jeremy Ralph Christian Roy Bobby "
    "Teresa Austin Carl Katherine Arthur Denise Lawrence Austin Roger "
    "Gerald Jane Ethan Ann Keith Jeremy Wayne Judy Gabriel Dylan Roy "
    "Juan Louis Diane Ralph Joyce Eugene Grace Gabriel Howard Ann "
    "Carl Russell Gloria Logan Mason Alan Wanda Randy Tracy Harry "
    "Kathryn Vincent Olivia Bobby Rose Phillip Victor Marie "
    "Bryan Kristen Leon Donna Howard Joy"
).split()

LAST_NAMES = (
    "Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez "
    "Hernandez Lopez Gonzalez Wilson Anderson Thomas Taylor Moore Jackson "
    "Martin Lee Perez Thompson White Harris Sanchez Clark Ramirez Lewis "
    "Robinson Walker Young Allen King Wright Scott Torres Nguyen Hill "
    "Flores Green Adams Nelson Baker Hall Rivera Campbell Mitchell Carter "
    "Roberts Gomez Phillips Evans Turner Diaz Parker Cruz Edwards Collins "
    "Reyes Stewart Morris Morales Murphy Cook Rogers Gutierrez Ortiz Morgan "
    "Cooper Peterson Bailey Reed Kelly Howard Ramos Kim Cox Ward Richardson "
    "Watson Brooks Chavez Wood James Bennett Gray Mendoza Ruiz Hughes Price "
    "Alvarez Castillo Myers Long Foster Jimenez Powell Jenkins Perry Russell "
    "Sullivan Bell Coleman Butler Henderson Barnes Gonzales Fisher Vasquez "
    "Simmons Romero Jordan Patterson Alexander Hamilton Graham Reynolds "
    "Griffin Wallace Moreno West Cole Hayes Bryant Herrera Gibson Ellis "
    "Tran Medina Aguilar Daniels Fernandez Wade Soto"
).split()

GAZETTEER_FIRST = FIRST_NAMES[:50]
GAZETTEER_LAST = LAST_NAMES[:50]
TRAIN_FIRST, TEST_FIRST = FIRST_NAMES[:100], FIRST_NAMES[100:150]
TRAIN_LAST, TEST_LAST = LAST_NAMES[:100], LAST_NAMES[100:150]

CITIES = [
    ("Springfield", "IL"), ("Riverside", "CA"), ("Franklin", "TN"),
    ("Georgetown", "TX"), ("Ashland", "OR"), ("Milford", "OH"),
    ("Clayton", "MO"), ("Bristol", "VA"), ("Dover", "DE"), ("Marion", "IA"),
]
STREETS = ["Main St", "Oak Ave", "Maple Rd", "Cedar Ln", "Elm St",
           "Washington Blvd", "Park Ave", "Lake Dr", "Hill Rd", "Church St"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

DISTRACTORS = [
    "BP 120/80, HR 72, Temp 98.6F, SpO2 98% on room air.",
    "Aspirin 81mg daily, Metformin 500mg BID, Lisinopril 10mg daily.",
    "Assessment: Type 2 diabetes mellitus, hypertension, hyperlipidemia.",
    "Plan: Continue current medications. Follow up with Cardiology in 2 weeks.",
    "Seen in the Emergency Department for evaluation.",
    "Admitted to the ICU for overnight observation.",
    "Labs: HbA1c 7.2%, LDL 110 mg/dL, creatinine 1.0.",
    "EKG shows normal sinus rhythm, no acute ST changes.",
    "Social history: non-smoker, occasional alcohol use.",
    "Review of systems negative except as noted above.",
    "Wound healing well, no signs of infection noted.",
    "Patient tolerated the procedure without complications.",
    "Discharge instructions provided and teach-back confirmed.",
    "Return precautions discussed in detail with the patient.",
]


def _rand_date(rng: random.Random, start_year: int, end_year: int) -> tuple[int, int, int]:
    year = rng.randint(start_year, end_year)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return year, month, day


def _fmt_date(rng: random.Random, ymd: tuple[int, int, int]) -> str:
    y, m, d = ymd
    style = rng.random()
    if style < 0.45:
        return f"{m:02d}/{d:02d}/{y}"
    if style < 0.7:
        return f"{MONTHS[m - 1]} {d}, {y}"
    if style < 0.85:
        return f"{m:02d}-{d:02d}-{y}"
    return f"{y}-{m:02d}-{d:02d}"


def _fmt_phone(rng: random.Random) -> str:
    a = rng.randint(200, 989)
    b = rng.randint(200, 989)
    c = rng.randint(1000, 9999)
    style = rng.random()
    if style < 0.4:
        return f"({a}) {b}-{c}"
    if style < 0.7:
        return f"{a}-{b}-{c}"
    if style < 0.9:
        return f"{a}.{b}.{c}"
    return f"+1 {a} {b} {c}"


def _fmt_address(rng: random.Random) -> str:
    num = rng.randint(101, 9899)
    street = rng.choice(STREETS)
    city, st = rng.choice(CITIES)
    z = rng.randint(10000, 99999)
    return f"{num} {street}, {city}, {st} {z}"


def _make_phi(rng: random.Random, first_pool, last_pool) -> dict:
    first = rng.choice(first_pool)
    last = rng.choice(last_pool)
    pfirst = rng.choice(first_pool)
    plast = rng.choice(last_pool)
    email_name = f"{first}.{last}".lower()
    return {
        "patient": f"{first} {last}",
        "patient_last": last,
        "provider": f"{pfirst} {plast}",
        "provider_dr": f"Dr. {pfirst} {plast}",
        "dob": _fmt_date(rng, _rand_date(rng, 1935, 2000)),
        "visit": _fmt_date(rng, _rand_date(rng, 2023, 2026)),
        "phone": _fmt_phone(rng),
        "mrn": str(rng.randint(10**6, 10**8 - 1)),
        "address": _fmt_address(rng),
        "email": f"{email_name}@example.com",
    }


# Each template is a list of segments: (text, entity_key or None).
# entity_key refers to a key of the phi dict produced by _make_phi.
def _templates(phi: dict) -> list[list[tuple[str, str | None]]]:
    p = phi
    return [
        [("Discharge Summary\nPatient: ", None), (p["patient"], "patient"),
         ("\nDOB: ", None), (p["dob"], "dob"),
         ("\nMRN: ", None), (p["mrn"], "mrn"), ("\n", None),
         ("Attending: ", None), (p["provider_dr"], "provider"),
         ("\nAdmission: ", None), (p["visit"], "visit"),
         ("\nPatient tolerated the procedure well. Discharge to home.\n", None)],
        [("Progress Note — ", None), (p["visit"], "visit"),
         ("\n", None), (p["patient"], "patient"),
         (" presents for follow-up. Seen by ", None), (p["provider_dr"], "provider"),
         (". Vitals stable. Continue current plan.\n", None)],
        [("CT Head without contrast.\nOrdering provider: ", None), (p["provider"], "provider"),
         ("\nPatient: ", None), (p["patient"], "patient"),
         ("  DOB ", None), (p["dob"], "dob"),
         ("\nFindings: No acute intracranial abnormality.\n", None)],
        [("Refill request for ", None), (p["patient"], "patient"),
         (". Best contact: ", None), (p["phone"], "phone"),
         (" or ", None), (p["email"], "email"),
         (". Home address on file: ", None), (p["address"], "address"), (".\n", None)],
        [("Lab Results\n", None), (p["patient"], "patient"),
         (" | DOB: ", None), (p["dob"], "dob"),
         (" | MRN ", None), (p["mrn"], "mrn"),
         ("\nHbA1c 7.2%, LDL 110. Results called to ", None), (p["phone"], "phone"), (".\n", None)],
        [("Dear ", None), (p["provider_dr"], "provider"),
         (",\nThank you for referring ", None), (p["patient"], "patient"),
         (" (DOB ", None), (p["dob"], "dob"),
         ("). We will schedule the consultation and notify your office.\n", None)],
        [("Telephone encounter: ", None), (p["patient"], "patient"),
         (" called regarding medication question. Callback number ", None),
         (p["phone"], "phone"), (". Spoke with patient directly.\n", None)],
        [("Operative Note\nSurgeon: ", None), (p["provider"], "provider"),
         ("\nPatient: ", None), (p["patient"], "patient"),
         ("\nDate: ", None), (p["visit"], "visit"),
         ("\nProcedure completed without complications.\n", None)],
        [("ED Course: ", None), (p["patient"], "patient"),
         (" arrived on ", None), (p["visit"], "visit"),
         (" with chest discomfort. Emergency contact: ", None), (p["phone"], "phone"),
         (". Workup negative. Discharged home.\n", None)],
        [("Follow-up visit ", None), (p["visit"], "visit"),
         (" with ", None), (p["provider_dr"], "provider"),
         (".\nMr. " if p["patient"].split()[0] in ("James", "John") else ".\nPatient ", None),
         (p["patient_last"], "patient"),
         (" reports improvement. Return in 3 months.\n", None)],
    ]


# Map phi-dict keys to canonical entity types.
ENTITY_TYPES = {
    "patient": "PATIENT", "provider": "PROVIDER", "dob": "DATE",
    "visit": "DATE", "phone": "PHONE", "mrn": "MRN",
    "address": "ADDRESS", "email": "EMAIL",
}


def generate_notes(n: int, seed: int, first_pool, last_pool) -> list[dict]:
    """Generate n synthetic notes. Returns [{text, entities:[{start,end,type,text}]}]."""
    rng = random.Random(seed)
    notes = []
    for _ in range(n):
        phi = _make_phi(rng, first_pool, last_pool)
        template = rng.choice(_templates(phi))
        parts: list[str] = []
        entities: list[dict] = []
        for text, key in template:
            start = sum(len(s) for s in parts)
            parts.append(text)
            if key is not None:
                entities.append({
                    "start": start,
                    "end": start + len(text),
                    "type": ENTITY_TYPES[key],
                    "text": text,
                })
        # 2-4 clinical distractor sentences (no PHI)
        for _ in range(rng.randint(2, 4)):
            parts.append(rng.choice(DISTRACTORS) + "\n")
        notes.append({"text": "".join(parts), "entities": entities})
    return notes


def generate_dataset(n_notes: int = N_NOTES, seed: int = SEED,
                     train_frac: float = TRAIN_FRAC) -> dict:
    n_train = int(n_notes * train_frac)
    train = generate_notes(n_train, seed, TRAIN_FIRST, TRAIN_LAST)
    test = generate_notes(n_notes - n_train, seed + 1000, TEST_FIRST, TEST_LAST)
    return {"train": train, "test": test,
            "meta": {"n_notes": n_notes, "seed": seed, "train_frac": train_frac,
                     "synthetic": True}}


def save_dataset(dataset: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataset, f)


def load_dataset(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    ds = generate_dataset()
    print(f"train={len(ds['train'])} test={len(ds['test'])}")
    print(ds["train"][0]["text"][:400])
    print(ds["train"][0]["entities"][:3])
