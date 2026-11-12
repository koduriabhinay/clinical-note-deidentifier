"""Redact one clinical note from the command line."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference import load_artifacts, redact_note

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    artifacts = load_artifacts()
    redacted, entities = redact_note(text, artifacts)
    print(redacted)
    print(f"\n[{len(entities)} entities redacted]")
    for e in entities:
        print(f"  {e['type']:9s} {e['text']!r} ({e['source']})")
