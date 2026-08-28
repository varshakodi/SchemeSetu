"""SchemeSetu — Week 1: a complete RAG pipeline built from scratch. No frameworks.

WHAT IS RAG (Retrieval-Augmented Generation)?
An LLM answering from memory will confidently invent scheme rules — that is
called *hallucination*. RAG fixes this by splitting the job in two:
  1. RETRIEVAL  — find the paragraphs in OUR documents most relevant to the
                  question (search problem, no AI generation involved).
  2. GENERATION — hand only those paragraphs to the LLM and instruct it to
                  answer strictly from them, with citations ("grounding").

The pipeline in this file:

    documents ──> chunk ──> embed ──> index          (run once:  `index`)
    question  ──> embed ──> compare with index ──> top chunks ──> LLM answer
                                                     (each time: `ask`)

Everything a "vector database" or LangChain does for you later in this
project, this file does by hand — so that in an interview, "what does
LangChain actually do?" has an answer you have personally written.

Usage:
    python naive/rag.py index                 # build the index (run after adding docs)
    python naive/rag.py ask "your question"   # retrieve (and answer, if API key set)

Retrieval works fully offline. Generating a final answer needs an Anthropic
API key:  export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# --- Configuration ------------------------------------------------------------
# Every value here is an *experiment knob*. When we change one, we re-run the
# evals and record the before/after numbers in decisions.md. That habit is the
# difference between "I used RAG" and "I engineered a retrieval system".

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `ingest` imports work when run as a script

from ingest.chunkers import CHUNKERS, load_documents  # noqa: E402

INDEX_FILE = Path(__file__).resolve().parent / "index.npz"
CHUNKS_FILE = Path(__file__).resolve().parent / "chunks.json"

# Embedding model. all-MiniLM-L6-v2 is small (~90 MB), fast on CPU, English-only.
# Week 6 we swap this single line to "BAAI/bge-m3" (multilingual, ~2 GB) for
# Hindi — and measure what that changes. See decisions.md entry 002.
EMBED_MODEL = "all-MiniLM-L6-v2"

# Chunking lives in ingest/chunkers.py (loading + cleaning + 3 strategies).
# "structure" won the Week-2 head-to-head — decisions.md entry 006 has the numbers.
CHUNK_STRATEGY = "structure"
TOP_K = 4              # how many chunks we retrieve for the LLM

# Retrieval mode: "dense" (embeddings only), "bm25" (keywords only), or
# "hybrid" (both, fused). Hybrid won Week 3 on hit@5 (the recall metric a
# downstream LLM cannot recover from) — decisions.md 009 has the numbers.
RETRIEVAL_MODE = "hybrid"
RRF_K = 60             # Reciprocal Rank Fusion constant (standard value)

# Reranking (Week 4). Embedding retrieval is a "bi-encoder": query and chunks
# are embedded INDEPENDENTLY, so similarity is approximate. A CROSS-encoder
# reads (query, chunk) together as one input and scores that exact pair —
# far more precise, far too slow to run on every chunk. Two-stage pattern:
# hybrid retrieval nominates RERANK_POOL candidates (recall), the
# cross-encoder re-orders them (precision). ms-marco-MiniLM is small (~80 MB)
# and English-only — swapped for BAAI/bge-reranker-v2-m3 in Week 6 alongside
# the embedder (same play as decisions.md 002).
RERANK = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 20

# Refusal (Week 4): if even the BEST reranked chunk scores below this, the
# corpus does not contain the answer — refuse instead of letting an LLM
# improvise. Tuned on dev-set traps; method and margin in decisions.md 011.
# Scale: cross-encoder logits (unbounded), not cosine.
REFUSAL_THRESHOLD = 1.5  # midpoint-ish of the dev gap: best trap +0.48,
                         # next answerable +3.10 (decisions.md 011; the one
                         # known false refusal, dev-012, is Week 5's job)

# Generation model — Claude Haiku 4.5: the cheap/fast tier, good grounding.
# Chosen in PRD §8; per-query cost matters for a student project.
GEN_MODEL = "claude-haiku-4-5"


# --- Steps 1 & 2: loading + chunking — now in ingest/chunkers.py --------------
# Week 1 implemented fixed-size chunking right here. Week 2 turned it into a
# three-way experiment: the original code lives on as ingest.chunkers
# .chunk_fixed (the baseline), alongside recursive and structure-aware
# strategies, raced head-to-head by evals/compare_chunkers.py.


# --- Step 3: embeddings -------------------------------------------------------

_embedder = None


def get_embedder():
    """Load the embedding model once, then reuse it (~90 MB download on first run).

    WHAT IS AN EMBEDDING? A function that turns text into a list of numbers
    (here: 384 of them — a "vector") such that texts with similar MEANING get
    nearby vectors. "Am I eligible?" and "Who qualifies?" share almost no
    words, but their vectors point in nearly the same direction. That is what
    lets retrieval beat plain keyword search.

    The model takes seconds to load, so we cache it in a module-level variable:
    without this, an eval over 55 questions would reload it 55 times. General
    rule: objects that are expensive to create and cheap to reuse get cached.
    """
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # slow import, done once
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def build_index() -> None:
    """Chunk every document, embed every chunk, save vectors + chunk texts to disk.

    We pass normalize_embeddings=True so every vector has length 1. Geometry
    payoff: for unit-length vectors, the dot product of two vectors IS their
    cosine similarity (how small the angle between them is, 1.0 = identical
    direction). So at query time, similarity search becomes one line of NumPy.
    """
    docs = load_documents()
    if not docs:
        sys.exit("No documents found. Add .md/.txt files to data/samples or data/raw first.")

    chunk_fn = CHUNKERS[CHUNK_STRATEGY]
    records = []
    for doc_id, title, text in docs:
        for i, chunk in enumerate(chunk_fn(text, title=title)):
            records.append({"doc_id": doc_id, "chunk_id": f"{doc_id}#{i}", "text": chunk})

    print(f"Embedding {len(records)} chunks from {len(docs)} documents "
          f"with {EMBED_MODEL} ({CHUNK_STRATEGY} chunking) ...")
    embedder = get_embedder()
    vectors = embedder.encode([r["text"] for r in records],
                              normalize_embeddings=True, show_progress_bar=True)

    np.savez(INDEX_FILE, vectors=vectors.astype(np.float32))
    CHUNKS_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=1))
    print(f"Saved index: {INDEX_FILE.name} ({vectors.shape[0]} vectors x "
          f"{vectors.shape[1]} dims), chunks: {CHUNKS_FILE.name}")


# --- Step 4: retrieval --------------------------------------------------------

_bm25 = None


def _get_bm25(records: list[dict]):
    """Build the BM25 index over chunk texts once per process, then reuse it."""
    global _bm25
    if _bm25 is None:
        from naive.bm25 import BM25
        _bm25 = BM25([r["text"] for r in records])
    return _bm25


_reranker = None


def _get_reranker():
    """Load the cross-encoder once per process (~80 MB download on first run)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def search(query: str, k: int = TOP_K, mode: str | None = None,
           rerank: bool | None = None) -> list[dict]:
    """Return the k best chunks for the query, by one of three retrieval modes.

    dense   Embed the query; rank chunks by cosine similarity (`vectors @ q`
            is a brute-force dot product against every chunk — a vector
            database like Qdrant is the scale-out version of this line).
            Strong on paraphrase ("money for my baby" ~ "maternity benefit"),
            weak on names and abbreviations.

    bm25    Rank by keyword relevance (see naive/bm25.py). Strong on exact
            names ("NMMSS", "PMSBY premium"), blind to meaning.

    hybrid  Run both, then merge with Reciprocal Rank Fusion (RRF): each
            chunk earns 1/(60 + rank) from each list that ranked it, and we
            sort by the sum. We fuse RANKS, not scores, because the two score
            scales are incomparable (cosine lives in [-1, 1]; BM25 is
            unbounded) — rank is the only common currency. A chunk that both
            retrievers like rises to the top; a chunk only one retriever
            loves still makes the list.

    With rerank on (the default), the mode above only NOMINATES a pool of
    RERANK_POOL candidates; the cross-encoder then re-scores each (query,
    chunk) pair and the top k by that score are returned, each carrying a
    `rerank` field. `cosine` is always present (comparable across configs).
    """
    mode = mode or RETRIEVAL_MODE
    rerank = RERANK if rerank is None else rerank
    if not INDEX_FILE.exists():
        sys.exit("No index found. Run:  python naive/rag.py index")

    vectors = np.load(INDEX_FILE)["vectors"]
    records = json.loads(CHUNKS_FILE.read_text())

    q = get_embedder().encode([query], normalize_embeddings=True)[0]
    cosine = vectors @ q
    dense_rank = list(np.argsort(cosine)[::-1])

    pool_n = max(k, RERANK_POOL) if rerank else k

    if mode == "dense":
        order, mode_scores = dense_rank[:pool_n], cosine
    else:
        bm25_scores = np.array(_get_bm25(records).scores(query))
        bm25_rank = list(np.argsort(bm25_scores)[::-1])
        if mode == "bm25":
            order, mode_scores = bm25_rank[:pool_n], bm25_scores
        elif mode == "hybrid":
            fused: dict[int, float] = {}
            for ranking in (dense_rank[:50], bm25_rank[:50]):
                for rank, idx in enumerate(ranking):
                    fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
            order = sorted(fused, key=fused.get, reverse=True)[:pool_n]
            mode_scores = fused
        else:
            sys.exit(f"Unknown retrieval mode: {mode!r}")

    if rerank:
        pool = order[:RERANK_POOL]
        pair_scores = _get_reranker().predict(
            [(query, records[int(i)]["text"]) for i in pool])
        ranked = sorted(zip(pool, pair_scores), key=lambda t: -t[1])[:k]
        return [{**records[int(i)], "score": float(s), "rerank": float(s),
                 "cosine": float(cosine[int(i)])} for i, s in ranked]

    return [{**records[int(i)], "score": float(mode_scores[int(i)]),
             "cosine": float(cosine[int(i)])} for i in order[:k]]


# --- Step 5: grounded generation ----------------------------------------------

SYSTEM_PROMPT = """You are SchemeSetu, an assistant for Indian government schemes.
Answer ONLY from the numbered context passages provided. After each factual
claim, cite the passage it came from, like [1] or [2].
If the passages do not contain the answer, say plainly that you don't know and
suggest what the user could ask instead. Never guess or use outside knowledge.
Answer in the language the question was asked in (an English question gets an
English answer, a Hindi question gets a Hindi answer), regardless of the
language of the scheme names involved."""


def build_context(hits: list[dict]) -> str:
    """Number the retrieved chunks — the numbering is what citations refer to."""
    return "\n\n".join(f"[{i + 1}] (source: {h['doc_id']})\n{h['text']}"
                       for i, h in enumerate(hits))


def generate_answer(query: str, hits: list[dict]) -> str:
    """Ask Claude to answer strictly from the retrieved chunks.

    This instruction pattern is called GROUNDING: the model is confined to the
    evidence we hand it, must cite it, and must refuse when the evidence is
    missing. Refusing correctly is a feature — we measure it in the evals
    ("trap refusal rate") because a wrong eligibility answer costs a citizen
    real time and money.
    """
    from agent.llm import get_llm  # lazy: retrieval-only use needs no provider

    return get_llm().complete(
        SYSTEM_PROMPT,
        f"Context passages:\n\n{build_context(hits)}\n\nQuestion: {query}")


# --- CLI ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="chunk + embed all documents, save the index")
    ask = sub.add_parser("ask", help="retrieve chunks for a question (and answer, if key set)")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default=None,
                     help=f"retrieval mode (default: {RETRIEVAL_MODE})")
    args = parser.parse_args()

    if args.command == "index":
        build_index()
        return

    hits = search(args.question, mode=args.mode)
    print(f"\nTop {len(hits)} retrieved chunks for: {args.question!r}\n" + "-" * 60)
    for i, h in enumerate(hits):
        preview = " ".join(h["text"].split())[:160]
        print(f"[{i+1}] score={h['score']:.3f} cos={h['cosine']:.3f}  {h['doc_id']}\n    {preview}...\n")

    if hits and hits[0].get("rerank") is not None and hits[0]["rerank"] < REFUSAL_THRESHOLD:
        print("-" * 60)
        print(f"REFUSING to answer: the best evidence scores {hits[0]['rerank']:.2f}, "
              f"below the refusal threshold ({REFUSAL_THRESHOLD:.2f}).")
        print("The corpus most likely does not contain this answer — refusing is a")
        print("feature, measured as 'trap refusal rate' in evals/results.md.")
        return

    from agent.llm import available_provider
    if available_provider():
        print("-" * 60 + "\nAnswer:\n")
        print(generate_answer(args.question, hits))
        print("\nSources:")
        for i, h in enumerate(hits):
            print(f"  [{i + 1}] {h['chunk_id']}")
    else:
        print("-" * 60)
        print("(Retrieval only — no LLM provider configured. Free options and\n"
              " setup: see agent/llm.py — Groq/Gemini keys need no card.)")


if __name__ == "__main__":
    main()
