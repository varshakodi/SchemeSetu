"""SchemeSetu — retrieval eval harness. Turns "it seems better" into numbers.

Runs every question in an eval file through retrieval and computes:

  hit@5   — For what fraction of questions does a chunk from a *gold document*
            (the doc we know contains the answer) appear in the top 5 results?
            The single most important retrieval number: if the right document
            isn't retrieved, no LLM downstream can answer correctly.

  MRR@10  — Mean Reciprocal Rank. For each question, take the rank of the
            FIRST gold-document chunk (1st -> 1.0, 2nd -> 0.5, 3rd -> 0.33...,
            not in top 10 -> 0), then average. hit@5 asks "did we find it?";
            MRR asks "how high did we rank it?" — both matter because the LLM
            pays most attention to the top of the context.

  Trap score gap — For trap questions (unanswerable from the corpus) there is
            no gold document, so instead we record the top-1 similarity score.
            If answerable questions score high and traps score low, a simple
            threshold can refuse traps — that gap is the raw material for
            Week 4's refusal logic. No LLM needed for any of this.

LLM-dependent metrics (faithfulness, answer relevance, refusal behaviour of
the *generated* answer) need an API key and an LLM judge — added later.

Usage:
    python evals/run_eval.py                      # golden set + dev set
    python evals/run_eval.py evals/dev_set.jsonl  # a specific file
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from naive.rag import search  # noqa: E402  (import after sys.path fix)

K_HIT = 5
K_MRR = 10


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(path: Path) -> None:
    rows = load_jsonl(path)
    answerable = [r for r in rows if r.get("gold_doc_ids")]
    traps = [r for r in rows if not r.get("gold_doc_ids")]

    print(f"\n=== {path.name}: {len(answerable)} answerable, {len(traps)} traps ===")

    hits, rranks, ans_top_scores = 0, [], []
    for r in answerable:
        results = search(r["question"], k=K_MRR)
        doc_ids = [h["doc_id"] for h in results]
        ans_top_scores.append(results[0]["score"])

        gold_rank = next((i + 1 for i, d in enumerate(doc_ids)
                          if d in r["gold_doc_ids"]), None)
        hit = gold_rank is not None and gold_rank <= K_HIT
        hits += hit
        rranks.append(1.0 / gold_rank if gold_rank else 0.0)
        mark = "HIT " if hit else "MISS"
        print(f"  [{mark}] rank={gold_rank or '>10'}  {r['id']}  {r['question'][:60]}")

    trap_top_scores = []
    for r in traps:
        results = search(r["question"], k=1)
        trap_top_scores.append(results[0]["score"])
        print(f"  [TRAP] top1_score={results[0]['score']:.3f}  {r['id']}  {r['question'][:60]}")

    print(f"\n  hit@{K_HIT}  = {hits}/{len(answerable)} = {hits / len(answerable):.2f}"
          if answerable else "  (no answerable questions)")
    if rranks:
        print(f"  MRR@{K_MRR} = {sum(rranks) / len(rranks):.3f}")
    if ans_top_scores:
        print(f"  avg top-1 score, answerable = {sum(ans_top_scores) / len(ans_top_scores):.3f}")
    if trap_top_scores:
        print(f"  avg top-1 score, traps      = {sum(trap_top_scores) / len(trap_top_scores):.3f}"
              f"   <- the gap vs answerable feeds Week 4's refusal threshold")


def main() -> None:
    files = [Path(a) for a in sys.argv[1:]] or [ROOT / "evals" / "golden_set.jsonl",
                                               ROOT / "evals" / "dev_set.jsonl"]
    for f in files:
        evaluate(f)


if __name__ == "__main__":
    main()
