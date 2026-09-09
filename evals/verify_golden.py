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
from pathlib import Path

DEFAULT_FILE = "evals/golden_candidates.jsonl"
TYPES = {"factual", "synthesis", "hindi", "trap", "ambiguous"}
PRD_MINIMUM = {"factual": 20, "synthesis": 10, "hindi": 10, "trap": 10, "ambiguous": 5}
# Word-overlap above this with any dev question counts as the same question.
OVERLAP_LIMIT = 0.5

# Subjects the traps ask about. Each must stay absent from the corpus.
TRAP_SUBJECTS = {
    "g-051": [r"mudra", r"\bPMMY\b"],
    "g-052": [r"svanidhi", r"street vendor"],
    "g-053": [r"surya ghar", r"solar", r"rooftop"],
    "g-054": [r"sukanya"],
    "g-055": [r"startup india", r"tax exemption"],
    "g-056": [r"vishwakarma", r"toolkit"],
    "g-057": [r"e-?shram"],
    "g-058": [r"\bABHA\b", r"digital mission", r"health id"],
    "g-059": [r"saubhagya"],
    "g-060": [r"ladli|laadli|behna"],
}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[\w\u0900-\u097F]+", text.lower().replace(",", "")))


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
        # Token-exact, not substring: "10" appears inside "10,000", so a
        # substring test passed a figure the document never stated -- it said
        # "ten years" in words. Compare whole numbers against whole numbers.
        text = " ".join(corpus[d] for d in r["gold_doc_ids"]).replace(",", "")
        present = set(re.findall(r"\d+", text))
        for num in {n.replace(",", "") for n in re.findall(r"\d[\d,]*", r["answer"])}:
            if len(num) >= 2 and num not in present:
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


def check_no_dev_overlap(rows: list[dict], fail) -> None:
    """No golden question may duplicate a dev-set question.

    The dev set is what the system was tuned against -- chunking, rerank pool,
    refusal threshold. A golden question that also appears there is measuring
    a question the system was already optimised for, which is the exact
    leakage the two-set split exists to prevent.

    This check exists because it happened: 16 of the first 68 golden questions
    were near-duplicates of dev questions, 8 of them word-for-word. Writing
    questions "from the corpus" is not enough -- the same corpus produces the
    same obvious questions twice.
    """
    dev_path = Path(__file__).with_name("dev_set.jsonl")
    if not dev_path.exists():
        return
    dev = [json.loads(line) for line in open(dev_path, encoding="utf-8")]
    dev_words = [_words(d["question"]) for d in dev]
    for r in rows:
        gw = _words(r["question"])
        for d, dw in zip(dev, dev_words):
            overlap = len(gw & dw) / max(1, len(gw | dw))
            if overlap >= OVERLAP_LIMIT:
                fail(f"{r['id']}: {overlap:.0%} word overlap with dev question "
                     f"{d['id']} -- the system was tuned on that. Rewrite it.")
                break


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
        ("no overlap with the dev set", lambda: check_no_dev_overlap(rows, fail)),
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
