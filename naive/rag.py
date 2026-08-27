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

def search(query: str, k: int = TOP_K) -> list[dict]:
    """Embed the question and return the k most similar chunks.

    `vectors @ q` computes the dot product of the query against EVERY chunk
    vector at once — brute force over the whole corpus. That's fine for a few
    hundred chunks. Making this fast at millions of vectors (and adding
    filters, persistence, sharding) is precisely the job of a vector database
    like Qdrant — which is why we adopt one in Week 3, and not before.
    """
    if not INDEX_FILE.exists():
        sys.exit("No index found. Run:  python naive/rag.py index")

    vectors = np.load(INDEX_FILE)["vectors"]
    records = json.loads(CHUNKS_FILE.read_text())

    q = get_embedder().encode([query], normalize_embeddings=True)[0]
    scores = vectors @ q                       # cosine similarity to every chunk
    top = np.argsort(scores)[::-1][:k]         # indices of the k best scores
    return [{**records[i], "score": float(scores[i])} for i in top]


# --- Step 5: grounded generation ----------------------------------------------

SYSTEM_PROMPT = """You are SchemeSetu, an assistant for Indian government schemes.
Answer ONLY from the numbered context passages provided. After each factual
claim, cite the passage it came from, like [1] or [2].
If the passages do not contain the answer, say plainly that you don't know and
suggest what the user could ask instead. Never guess or use outside knowledge.
Answer in the same language as the question."""


def generate_answer(query: str, hits: list[dict]) -> str:
    """Ask Claude to answer strictly from the retrieved chunks.

    This instruction pattern is called GROUNDING: the model is confined to the
    evidence we hand it, must cite it, and must refuse when the evidence is
    missing. Refusing correctly is a feature — we measure it in the evals
    ("trap refusal rate") because a wrong eligibility answer costs a citizen
    real time and money.
    """
    import anthropic  # imported lazily so retrieval-only use needs no API key

    context = "\n\n".join(f"[{i+1}] (source: {h['doc_id']})\n{h['text']}"
                          for i, h in enumerate(hits))
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=GEN_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user",
                   "content": f"Context passages:\n\n{context}\n\nQuestion: {query}"}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# --- CLI ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="chunk + embed all documents, save the index")
    ask = sub.add_parser("ask", help="retrieve chunks for a question (and answer, if key set)")
    ask.add_argument("question")
    args = parser.parse_args()

    if args.command == "index":
        build_index()
        return

    hits = search(args.question)
    print(f"\nTop {len(hits)} retrieved chunks for: {args.question!r}\n" + "-" * 60)
    for i, h in enumerate(hits):
        preview = " ".join(h["text"].split())[:160]
        print(f"[{i+1}] score={h['score']:.3f}  {h['doc_id']}\n    {preview}...\n")

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("-" * 60 + "\nAnswer:\n")
        print(generate_answer(args.question, hits))
    else:
        print("-" * 60)
        print("(Retrieval only — no ANTHROPIC_API_KEY set. Export one to get a\n"
              " grounded, cited answer generated from the chunks above.)")


if __name__ == "__main__":
    main()
