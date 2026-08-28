"""SchemeSetu — Week 3 experiment: dense vs BM25 vs hybrid, per-question ranks.

The comparison that matters this week is not one aggregate number but the
per-question rank matrix: WHERE does each retriever fail? The theory says
dense embeddings stumble on names/abbreviations/terse keyword queries, BM25
stumbles on paraphrases with no word overlap, and hybrid (RRF fusion) covers
both. The matrix shows whether our corpus agrees with the theory.

Usage:  python evals/compare_retrieval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from naive.rag import search  # noqa: E402

MODES = ["dense", "bm25", "hybrid"]
K_HIT, K_MRR = 5, 10


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def first_gold_rank(results: list[dict], gold: list[str]) -> int | None:
    for i, h in enumerate(results):
        if h["doc_id"] in gold:
            return i + 1
    return None


def evaluate(path: Path) -> None:
    rows = load_jsonl(path)
    answerable = [r for r in rows if r.get("gold_doc_ids")]
    traps = [r for r in rows if not r.get("gold_doc_ids")]
    print(f"\n=== {path.name} — rank of first gold doc per mode (>{K_MRR} = not found) ===")
    print(f"{'id':<9} {'type':<11} {'dense':>6} {'bm25':>6} {'hybrid':>7}   question")

    agg = {m: {"hits": 0, "rr": []} for m in MODES}
    for r in answerable:
        ranks = {}
        for m in MODES:
            res = search(r["question"], k=K_MRR, mode=m)
            rank = first_gold_rank(res, r["gold_doc_ids"])
            ranks[m] = rank
            agg[m]["hits"] += bool(rank and rank <= K_HIT)
            agg[m]["rr"].append(1.0 / rank if rank else 0.0)
        marks = {m: (str(v) if v else f">{K_MRR}") for m, v in ranks.items()}
        flag = "  <-- disagreement" if len({v or 99 for v in ranks.values()}) > 1 else ""
        print(f"{r['id']:<9} {r.get('type', '?'):<11} {marks['dense']:>6} "
              f"{marks['bm25']:>6} {marks['hybrid']:>7}   {r['question'][:48]}{flag}")

    for r in traps:
        cos = {m: search(r["question"], k=1, mode=m)[0]["cosine"] for m in MODES}
        print(f"{r['id']:<9} {'TRAP':<11} {cos['dense']:>6.2f} {cos['bm25']:>6.2f} "
              f"{cos['hybrid']:>7.2f}   {r['question'][:48]}  (top-1 cosine)")

    n = len(answerable)
    print(f"\n{'mode':<8} {'hit@5':>7} {'MRR@10':>8}")
    for m in MODES:
        print(f"{m:<8} {agg[m]['hits'] / n:>7.2f} {sum(agg[m]['rr']) / n:>8.3f}")


def main() -> None:
    for name in ("dev_set.jsonl", "golden_set.jsonl"):
        evaluate(ROOT / "evals" / name)


if __name__ == "__main__":
    main()
