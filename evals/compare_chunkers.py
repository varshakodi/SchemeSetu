"""SchemeSetu — Week 2 experiment: race the chunking strategies, dev set decides.

Four configurations run under identical conditions:

  fixed-raw      Week 1's exact pipeline (fixed chunks, uncleaned docs) — the
                 replica of the recorded phase-1 baseline.
  fixed-clean    Same chunker, but on cleaned documents. Isolates how much of
                 any improvement comes from cleaning alone ("change one
                 variable at a time" — otherwise we can't attribute wins).
  recursive      Recursive splitting on cleaned docs.
  structure      Structure-aware chunks with [Title › Section] labels, cleaned.

For each config we build an in-memory index (nothing on disk is touched) and
score the dev set: hit@5, MRR@10, average top-1 similarity for answerable
questions vs traps, plus a hygiene stat — the share of chunks that start with
a lowercase letter (a cheap fingerprint for "this chunk begins mid-word or
mid-sentence"). The golden set is scored too, for reference only — decisions
are made on the dev set; the golden set is never tuned against.

Usage:  python evals/compare_chunkers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest.chunkers import CHUNKERS, load_documents  # noqa: E402
from naive.rag import get_embedder  # noqa: E402

K_HIT, K_MRR = 5, 10

CONFIGS = [
    ("fixed-raw", "fixed", False),
    ("fixed-clean", "fixed", True),
    ("recursive", "recursive", True),
    ("structure", "structure", True),
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_in_memory_index(strategy: str, clean: bool):
    chunk_fn = CHUNKERS[strategy]
    records = []
    for doc_id, title, text in load_documents(clean=clean):
        for chunk in chunk_fn(text, title=title):
            records.append({"doc_id": doc_id, "text": chunk})
    vectors = get_embedder().encode([r["text"] for r in records],
                                    normalize_embeddings=True, show_progress_bar=False)
    return vectors, records


def score(vectors, records, questions: list[dict]) -> dict:
    embedder = get_embedder()
    hits, rranks, ans_scores, trap_scores = 0, [], [], []
    misses = []
    for r in questions:
        q = embedder.encode([r["question"]], normalize_embeddings=True)[0]
        sims = vectors @ q
        top = np.argsort(sims)[::-1][:K_MRR]
        doc_ids = [records[i]["doc_id"] for i in top]
        if r.get("gold_doc_ids"):
            ans_scores.append(float(sims[top[0]]))
            rank = next((i + 1 for i, d in enumerate(doc_ids) if d in r["gold_doc_ids"]), None)
            if rank and rank <= K_HIT:
                hits += 1
            else:
                misses.append(f"{r['id']}(rank={rank or '>10'})")
            rranks.append(1.0 / rank if rank else 0.0)
        else:
            trap_scores.append(float(sims[top[0]]))
    n_ans = len(ans_scores)
    return {
        "hit": hits / n_ans if n_ans else 0.0,
        "mrr": sum(rranks) / len(rranks) if rranks else 0.0,
        "ans": sum(ans_scores) / n_ans if n_ans else 0.0,
        "trap": sum(trap_scores) / len(trap_scores) if trap_scores else 0.0,
        "misses": misses,
    }


def main() -> None:
    dev = load_jsonl(ROOT / "evals" / "dev_set.jsonl")
    gold = load_jsonl(ROOT / "evals" / "golden_set.jsonl")

    print(f"{'config':<12} {'chunks':>6} {'%lc-start':>9} | "
          f"{'dev hit@5':>9} {'dev MRR':>8} {'ans':>6} {'trap':>6} | {'gold hit':>8} {'gold MRR':>8}")
    print("-" * 96)

    example_target = "Means-cum-Merit"
    examples = {}

    for name, strategy, clean in CONFIGS:
        vectors, records = build_in_memory_index(strategy, clean)
        lc = sum(bool(re.match(r"^[a-z]", r["text"])) for r in records) / len(records)
        d = score(vectors, records, dev)
        g = score(vectors, records, gold)
        flag = f"   misses: {', '.join(d['misses'])}" if d["misses"] else ""
        print(f"{name:<12} {len(records):>6} {lc:>8.0%} | "
              f"{d['hit']:>9.2f} {d['mrr']:>8.3f} {d['ans']:>6.2f} {d['trap']:>6.2f} | "
              f"{g['hit']:>8.2f} {g['mrr']:>8.3f}{flag}")
        ex = next((r["text"] for r in records if example_target in r["text"]), None)
        if ex:
            examples[name] = " ".join(ex[:180].split())

    print("\nSample chunk containing 'Means-cum-Merit' per config (first 180 chars):")
    for name, ex in examples.items():
        print(f"\n[{name}]\n  {ex}")


if __name__ == "__main__":
    main()
