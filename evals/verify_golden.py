"""Tests for the answer key itself.

An eval set is data that decides whether the system passes. If the data is
wrong, every number downstream is wrong and looks fine. So the golden set
gets its own tests:

  1. structure   -- unique ids, required fields, known types
  2. gold docs   -- every cited filename exists in the corpus
  3. figures     -- every number in an answer appears in its gold documents
  4. traps       -- every trap subject is genuinely ABSENT from the corpus
  5. composition -- slice counts against the PRD targets

Check 4 is the one that rots. A trap is only a trap while the corpus cannot
answer it; the day someone ingests a Mudra document, g-051 quietly stops
testing refusal and starts testing retrieval. Re-run this after every
corpus change.

    python evals/verify_golden.py [--file evals/golden_set.jsonl]

Exits non-zero if any check fails, so it can gate a commit.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

DEFAULT_FILE = "evals/golden_candidates.jsonl"
TYPES = {"factual", "synthesis", "hindi", "trap", "ambiguous"}
PRD_MINIMUM = {"factual": 20, "synthesis": 10, "hindi": 10, "trap": 10, "ambiguous": 5}

# Subjects the traps ask about. Each must stay absent from the corpus.
TRAP_SUBJECTS = {
    "g-051": [r"mudra", r"\bPMMY\b"],
    "g-052": [r"\bFAME\b", r"electric (two|scooter|vehicle)"],
    "g-053": [r"PMAY[- ]?U\b", r"carpet area", r"urban housing"],
    "g-054": [r"sukanya"],
    "g-055": [r"startup india", r"tax exemption"],
    "g-056": [r"vishwakarma", r"toolkit"],
    "g-057": [r"e-?shram"],
    "g-058": [r"\bABHA\b", r"digital mission", r"health id"],
    "g-059": [r"drone"],
    "g-060": [r"ladli|laadli|behna"],
}


def load_corpus() -> dict[str, str]:
    return {
        os.path.basename(p): open(p, encoding="utf-8").read()
        for p in glob.glob("data/**/*.*", recursive=True)
        if p.endswith((".md", ".txt"))
    }


def check_structure(rows: list[dict], fail) -> None:
    seen: set[str] = set()
    for i, r in enumerate(rows):
        for field in ("id", "type", "language", "question", "answer", "gold_doc_ids"):
            if field not in r:
                fail(f"row {i} missing field '{field}'")
        if r.get("id") in seen:
            fail(f"duplicate id {r['id']}")
        seen.add(r.get("id"))
        if r.get("type") not in TYPES:
            fail(f"{r.get('id')}: unknown type {r.get('type')!r}")
        if r.get("type") in {"trap", "ambiguous"} and r.get("gold_doc_ids"):
            fail(f"{r['id']}: a {r['type']} question must have no gold documents")
        if r.get("type") not in {"trap", "ambiguous"} and not r.get("gold_doc_ids"):
            fail(f"{r['id']}: answerable question has no gold documents")


def check_gold_docs(rows: list[dict], corpus: dict[str, str], fail) -> None:
    for r in rows:
        for doc in r["gold_doc_ids"]:
            if doc not in corpus:
                fail(f"{r['id']}: gold document not in corpus: {doc}")


def check_figures(rows: list[dict], corpus: dict[str, str], fail) -> None:
    """Every figure in an answer must appear in a cited document.

    Catches outside knowledge leaking into the answer key -- a number that is
    true in the world but absent from the corpus makes the question
    ungradeable, because the system cannot cite what it was never given.
    """
    for r in rows:
        if not r["gold_doc_ids"]:
            continue
        text = " ".join(corpus[d] for d in r["gold_doc_ids"]).replace(",", "")
        for num in {n.replace(",", "") for n in re.findall(r"\d[\d,]*", r["answer"])}:
            if len(num) >= 2 and num not in text:
                fail(f"{r['id']}: figure {num} is not in {', '.join(r['gold_doc_ids'])}")


def check_traps(rows: list[dict], corpus: dict[str, str], fail) -> None:
    traps = {r["id"] for r in rows if r["type"] == "trap"}
    for tid in sorted(traps):
        if tid not in TRAP_SUBJECTS:
            fail(f"{tid}: trap has no absence patterns in TRAP_SUBJECTS -- add them")
            continue
        for pat in TRAP_SUBJECTS[tid]:
            for name, text in corpus.items():
                if re.search(pat, text, re.I):
                    fail(
                        f"{tid}: NO LONGER A TRAP -- /{pat}/ now appears in {name}. "
                        "Replace the question or drop it."
                    )


def check_composition(rows: list[dict], warn) -> None:
    counts = {t: sum(1 for r in rows if r["type"] == t) for t in sorted(TYPES)}
    for t, minimum in PRD_MINIMUM.items():
        if counts[t] < minimum:
            warn(f"{t}: {counts[t]} questions, PRD asks for {minimum}")
    print("  composition: " + " · ".join(f"{t} {n}" for t, n in counts.items())
          + f" · total {len(rows)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_FILE)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.file, encoding="utf-8")]
    corpus = load_corpus()
    failures: list[str] = []
    warnings: list[str] = []
    fail = failures.append
    warn = warnings.append

    print(f"Verifying {args.file} against {len(corpus)} corpus documents\n")
    for label, fn in (
        ("structure", lambda: check_structure(rows, fail)),
        ("gold documents exist", lambda: check_gold_docs(rows, corpus, fail)),
        ("figures supported", lambda: check_figures(rows, corpus, fail)),
        ("traps still unanswerable", lambda: check_traps(rows, corpus, fail)),
        ("composition", lambda: check_composition(rows, warn)),
    ):
        before = len(failures)
        fn()
        mark = "ok  " if len(failures) == before else "FAIL"
        print(f"  [{mark}] {label}")

    for w in warnings:
        print(f"\n  warning: {w}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
