"""SchemeSetu — Week 6 A/B: can retrieval cross the language barrier?

The corpus is English. A Hindi question shares ZERO tokens with it, which
kills keyword search outright and challenges any English-only embedder. Two
remedies compete (PRD FR-6.1), and this script referees:

  bge-hi        BGE-M3, a MULTILINGUAL embedder: Hindi and English meaning
                land at nearby points in ONE shared vector space, so a Hindi
                query can find an English chunk directly ("cross-lingual").
  translate-hi  Translate the question to English with the LLM first, then
                retrieve with the existing English embedder. Simpler idea,
                but adds an LLM call (latency + a failure point) per query.

Controls that make the comparison honest:
  minilm-hi     The current embedder on Hindi questions — the broken baseline.
  bm25-hi       Keyword search on Hindi questions — the expected zero, shown
                deliberately: hybrid fusion contributes NOTHING cross-lingual.
  minilm-en /   Both embedders on the ENGLISH questions — swapping the
  bge-en        embedder must not regress English (the check almost everyone
                forgets: a swap that wins Hindi but loses English is a trade,
                not an upgrade).

Retrieval is dense-only here (no reranker) — one variable at a time: this
experiment isolates the EMBEDDER. The reranker swap is a separate step with
its own threshold re-tuning (decisions 011/016).

Usage:  python evals/compare_embedders.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest.chunkers import CHUNKERS, load_documents  # noqa: E402
from naive.bm25 import BM25  # noqa: E402

K_HIT, K_MRR = 5, 10
MODELS = {"minilm": "all-MiniLM-L6-v2", "bge": "BAAI/bge-m3"}


def load_questions() -> tuple[list[dict], list[dict]]:
    rows = [json.loads(l) for l in (ROOT / "evals" / "dev_set.jsonl").read_text().splitlines() if l.strip()]
    answerable = [r for r in rows if r.get("gold_doc_ids")]
    return ([r for r in answerable if r.get("language") == "en"],
            [r for r in answerable if r.get("language") == "hi"])


def build_chunks() -> list[dict]:
    records = []
    for doc_id, title, text in load_documents(clean=True):
        for chunk in CHUNKERS["structure"](text, title=title):
            records.append({"doc_id": doc_id, "text": chunk})
    return records


def score_dense(vectors, embedder, records, questions, query_of) -> tuple[float, float]:
    hits, rranks = 0, []
    for r in questions:
        q = embedder.encode([query_of(r)], normalize_embeddings=True)[0]
        top = np.argsort(vectors @ q)[::-1][:K_MRR]
        doc_ids = [records[i]["doc_id"] for i in top]
        rank = next((i + 1 for i, d in enumerate(doc_ids) if d in r["gold_doc_ids"]), None)
        hits += bool(rank and rank <= K_HIT)
        rranks.append(1.0 / rank if rank else 0.0)
    n = len(questions)
    return hits / n, sum(rranks) / n


def score_bm25(records, questions) -> tuple[float, float]:
    bm25 = BM25([r["text"] for r in records])
    hits, rranks = 0, []
    for r in questions:
        scores = np.array(bm25.scores(r["question"]))
        top = np.argsort(scores)[::-1][:K_MRR]
        doc_ids = [records[i]["doc_id"] for i in top]
        rank = next((i + 1 for i, d in enumerate(doc_ids)
                     if d in r["gold_doc_ids"] and scores[top[i - 1]] > 0), None)
        hits += bool(rank and rank <= K_HIT)
        rranks.append(1.0 / rank if rank else 0.0)
    n = len(questions)
    return hits / n, sum(rranks) / n


def main() -> None:
    from sentence_transformers import SentenceTransformer

    en_q, hi_q = load_questions()
    records = build_chunks()
    texts = [r["text"] for r in records]
    print(f"{len(records)} chunks · {len(en_q)} EN questions · {len(hi_q)} HI questions")

    embedders, vectors = {}, {}
    for name, model_id in MODELS.items():
        print(f"embedding corpus with {model_id} ...")
        embedders[name] = SentenceTransformer(model_id)
        vectors[name] = embedders[name].encode(texts, normalize_embeddings=True,
                                               show_progress_bar=False)

    translations: dict[str, str] = {}

    def translated(r: dict) -> str:
        if r["id"] not in translations:
            from agent.llm import get_llm
            translations[r["id"]] = get_llm().complete(
                "Translate this question to English. Reply with the translation only.",
                r["question"], max_tokens=100)
        return translations[r["id"]]

    print(f"\n{'config':<14} {'questions':<10} {'hit@5':>6} {'MRR@10':>7}")
    print("-" * 42)
    rows = [
        ("minilm-en", en_q, lambda: score_dense(vectors["minilm"], embedders["minilm"], records, en_q, lambda r: r["question"])),
        ("bge-en", en_q, lambda: score_dense(vectors["bge"], embedders["bge"], records, en_q, lambda r: r["question"])),
        ("minilm-hi", hi_q, lambda: score_dense(vectors["minilm"], embedders["minilm"], records, hi_q, lambda r: r["question"])),
        ("bm25-hi", hi_q, lambda: score_bm25(records, hi_q)),
        ("bge-hi", hi_q, lambda: score_dense(vectors["bge"], embedders["bge"], records, hi_q, lambda r: r["question"])),
        ("translate-hi", hi_q, lambda: score_dense(vectors["minilm"], embedders["minilm"], records, hi_q, translated)),
    ]
    for name, qs, fn in rows:
        hit, mrr = fn()
        print(f"{name:<14} {len(qs):<10} {hit:>6.2f} {mrr:>7.3f}")

    if translations:
        print("\nLLM translations used by translate-hi:")
        for qid, t in translations.items():
            print(f"  {qid}: {t[:70]}")


if __name__ == "__main__":
    main()
