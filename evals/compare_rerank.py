"""SchemeSetu — Week 4 experiment: does reranking restore precision, and where
should the refusal threshold sit?

Part 1 — ranks. For every answerable question: rank of the first gold document
under hybrid alone vs hybrid + cross-encoder rerank. Week 3 bought recall
(hit@5 = 1.00) by diluting MRR to 0.917; the reranker's job is to push the
gold chunk back to rank 1 within the candidate pool.

Part 2 — threshold. For refusal we need a score that separates "the corpus
answers this" from "it doesn't". We collect the TOP-1 rerank score for every
answerable question and every trap (tuning on the DEV set only), print both
distributions, and suggest the midpoint of the gap between the worst
answerable score and the best trap score. The margin between those two numbers
is how much room the threshold has — a thin margin means refusal will be
fragile and needs more trap data (grow the golden set!).

Usage:  python evals/compare_rerank.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from naive.rag import search  # noqa: E402

K = 10


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def first_gold_rank(results: list[dict], gold: list[str]) -> int | None:
    for i, h in enumerate(results):
        if h["doc_id"] in gold:
            return i + 1
    return None


def evaluate(path: Path, collect: dict | None = None) -> None:
    rows = load_jsonl(path)
    answerable = [r for r in rows if r.get("gold_doc_ids")]
    traps = [r for r in rows if not r.get("gold_doc_ids")]

    print(f"\n=== {path.name} — rank of first gold doc (>{K} = not found) ===")
    print(f"{'id':<9} {'type':<11} {'hybrid':>7} {'+rerank':>8}   question")

    agg = {"before": [], "after": []}
    for r in answerable:
        before = first_gold_rank(search(r["question"], k=K, rerank=False), r["gold_doc_ids"])
        after_results = search(r["question"], k=K, rerank=True)
        after = first_gold_rank(after_results, r["gold_doc_ids"])
        agg["before"].append(1.0 / before if before else 0.0)
        agg["after"].append(1.0 / after if after else 0.0)
        if collect is not None:
            collect["ans"].append(after_results[0]["rerank"])
        flag = "  <-- changed" if before != after else ""
        print(f"{r['id']:<9} {r.get('type', '?'):<11} {before or f'>{K}':>7} "
              f"{after or f'>{K}':>8}   {r['question'][:44]}{flag}")

    for r in traps:
        top = search(r["question"], k=1, rerank=True)[0]["rerank"]
        if collect is not None:
            collect["trap"].append(top)
        print(f"{r['id']:<9} {'TRAP':<11} {'':>7} {top:>8.2f}   {r['question'][:44]}  (top-1 rerank)")

    n = len(answerable)
    print(f"\n  MRR@{K}: hybrid {sum(agg['before']) / n:.3f}  ->  +rerank {sum(agg['after']) / n:.3f}")


def main() -> None:
    dev_scores = {"ans": [], "trap": []}
    evaluate(ROOT / "evals" / "dev_set.jsonl", collect=dev_scores)
    evaluate(ROOT / "evals" / "golden_set.jsonl")  # reference only — never tuned on

    print("\n=== Refusal threshold analysis (dev set only) ===")
    print(f"  answerable top-1 rerank scores: {sorted(round(s, 2) for s in dev_scores['ans'])}")
    print(f"  trap       top-1 rerank scores: {sorted(round(s, 2) for s in dev_scores['trap'])}")
    lo, hi = min(dev_scores["ans"]), max(dev_scores["trap"])
    print(f"  worst answerable = {lo:.2f}   best trap = {hi:.2f}   margin = {lo - hi:.2f}")
    if lo > hi:
        print(f"  clean separation — suggested threshold (midpoint): {(lo + hi) / 2:.2f}")
    else:
        print("  DISTRIBUTIONS OVERLAP — a hard threshold will misfire; "
              "needs more trap data or a different signal.")


if __name__ == "__main__":
    main()
