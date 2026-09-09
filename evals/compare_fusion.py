"""Does RRF need to ignore a retriever that has no opinion?

Found by the frozen golden set: a Hindi question ("how much is the Haryana old
age allowance?") had its gold document at DENSE rank 1 and HYBRID rank 19 --
outside the rerank pool, so it was never scored and the question was missed.

Cause: BM25 is language-blind, and for a Devanagari query it scores all 132
chunks exactly 0.000. `np.argsort` on an all-zero array still returns an
order -- an arbitrary tie order -- and the fusion loop hands the first 50 of
those a real 1/(60+rank) vote. RRF cannot tell "no opinion" from "ranked
last", so noise outvoted a correct dense hit.

Guard tested here: a retriever only votes if it actually discriminated
(any non-zero score).

Run on the DEV set. The golden set is frozen; its numbers stand as measured.

    python evals/compare_fusion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from naive.rag import (  # noqa: E402
    CHUNKS_FILE, INDEX_FILE, RERANK_MAX_CHARS, RERANK_POOL, RRF_K,
    _get_bm25, _get_reranker, get_embedder,
)

K_HIT, K_MRR = 5, 10


def search_variant(query: str, records, vectors, k: int, guard: bool):
    q = get_embedder().encode([query], normalize_embeddings=True)[0]
    cosine = vectors @ q
    dense_rank = list(np.argsort(cosine)[::-1])
    bm25_scores = np.array(_get_bm25(records).scores(query))
    bm25_rank = list(np.argsort(bm25_scores)[::-1])

    rankings = [dense_rank[:50]]
    # The guard: an all-zero score vector is not a ranking, it is silence.
    if not guard or bm25_scores.max() > 0:
        rankings.append(bm25_rank[:50])

    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
    order = sorted(fused, key=fused.get, reverse=True)[:RERANK_POOL]

    pairs = [(query, records[int(i)]["text"][:RERANK_MAX_CHARS]) for i in order]
    scores = _get_reranker().predict(pairs)
    ranked = sorted(zip(order, scores), key=lambda t: -t[1])[:k]
    return [records[int(i)]["doc_id"] for i, _ in ranked]


def evaluate(rows, records, vectors, guard: bool):
    hits, rranks, per_lang = 0, [], {}
    answerable = [r for r in rows if r.get("gold_doc_ids")]
    for r in answerable:
        docs = search_variant(r["question"], records, vectors, K_MRR, guard)
        rank = next((i + 1 for i, d in enumerate(docs) if d in r["gold_doc_ids"]), None)
        hit = rank is not None and rank <= K_HIT
        hits += hit
        rranks.append(1.0 / rank if rank else 0.0)
        lang = r.get("language", "en")
        per_lang.setdefault(lang, [0, 0])
        per_lang[lang][0] += hit
        per_lang[lang][1] += 1
    return {
        "hit": hits / len(answerable),
        "mrr": sum(rranks) / len(rranks),
        "n": len(answerable),
        "by_lang": {k: v[0] / v[1] for k, v in per_lang.items()},
    }


def main() -> None:
    files = [Path(a) for a in sys.argv[1:]] or [ROOT / "evals" / "dev_set.jsonl"]
    records = json.loads(CHUNKS_FILE.read_text())
    vectors = np.load(INDEX_FILE)["vectors"]
    for path in files:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        print(f"\n=== {path.name} ===")
        print(f"{'fusion':<28}{'hit@5':>8}{'MRR@10':>9}   by language")
        for guard, label in ((False, "current (both always vote)"),
                             (True, "guarded (silence != vote)")):
            m = evaluate(rows, records, vectors, guard)
            langs = "  ".join(f"{k}={v:.2f}" for k, v in sorted(m["by_lang"].items()))
            print(f"{label:<28}{m['hit']:>8.2f}{m['mrr']:>9.3f}   {langs}")


if __name__ == "__main__":
    main()
