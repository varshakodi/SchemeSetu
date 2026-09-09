# SchemeSetu

**Ask about Indian government schemes in plain language — get answers grounded in official documents, with citations, or an honest "I don't know."**

[![CI](https://github.com/varshakodi/SchemeSetu/actions/workflows/ci.yml/badge.svg)](https://github.com/varshakodi/SchemeSetu/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Evals](https://img.shields.io/badge/evals-tracked_every_phase-orange.svg)](evals/results.md)

**▶ Try it live: [schemesetu.streamlit.app](https://schemesetu.streamlit.app)** — ask in English or Hindi.
Free hosting sleeps after ~12 hours idle, so the first visit takes a minute to wake, and an answer
takes ~22 s on a shared CPU (1.3 s locally — see [`evals/results.md`](evals/results.md) row 7).

SchemeSetu is a bilingual (English/Hindi) retrieval-augmented generation (RAG)
system over official Indian government scheme documents — PM-KISAN, PMAY-G,
Ayushman Bharat, scholarships, pensions and more. It is built
**evaluation-first**: every architectural choice (chunking strategy, retrieval
mode, reranking) is decided by measured before/after numbers on a frozen
question set, and every decision is recorded with its evidence in
[`decisions.md`](decisions.md).

## Why this exists

- **3,000+ schemes, buried in PDFs.** Eligibility rules, benefit amounts and
  document checklists live in circulars written in bureaucratic language,
  scattered across dozens of portals.
- **Wrong answers cost real money.** A citizen acting on a hallucinated
  eligibility rule wastes application fees and weeks of effort. Grounding,
  per-claim citations and *measured* refusal are therefore first-class
  requirements here — not nice-to-haves.
- **Existing tools don't cite.** myScheme offers browse/filter; per-scheme
  chatbots exist (PM-KISAN's assistant, Ayushman Sarathi). SchemeSetu differs
  in three ways: citations to the exact source passage, a tracked
  honest-refusal rate on out-of-corpus questions, and a public eval report.

## How it works

```mermaid
flowchart LR
    Q[Question] --> D[Dense retrieval\nBGE-M3 multilingual]
    Q --> B[BM25 keyword retrieval\nbuilt from scratch]
    D --> F[Reciprocal\nRank Fusion]
    B --> F
    F --> R[Cross-encoder rerank\ntop 16 to top 4]
    R --> T{Confidence\nthreshold}
    T -->|above| G[Claude generates\ncited answer]
    T -->|below| N[Honest refusal]
```

Every stage is implemented **from scratch first** (see [`naive/`](naive/) —
embedding search in plain NumPy, BM25 in ~60 lines) before any framework, so
the repository doubles as a working tutorial on how RAG actually works under
the hood. Frameworks (LangChain/LangGraph) enter only where they earn their
keep — see the roadmap.

## Evaluation-driven development

The headline numbers come from a **frozen golden set of 68 questions** — 33
factual, 10 multi-document synthesis, 10 Hindi, 10 out-of-corpus traps and 5
deliberately underspecified. It was written from the corpus, verified by its
own test suite, frozen, and run once.

| Metric | Result | Target (PRD §5) | |
|---|---|---|---|
| hit@5 | **0.96** (51/53) | ≥ 0.85 | ✓ |
| MRR@10 | **0.925** | — | |
| Trap refusal | **1.00** (10/10) | ≥ 0.90 | ✓ |
| Cites the gold document | 48/49 *(full set)* | — | |
| False refusal | 0.08 *(full set, agent-level)* | ≤ 0.10 | ✓ |
| Faithfulness | *not yet validly measured* | ≥ 0.90 | see below |

Measured on **golden v2** — 68 questions, none of which overlaps the dev set the
system was tuned on. v1 contained 16 duplicates and is withdrawn; see *What the
golden set caught*, below. Rows marked *(full set)* await a clean agent re-run.

Tuning happened on a separate dev set, never on these questions. That
separation is the whole point: numbers from a set you tuned against describe
how well the system does on questions it was already optimised for.

| Retrieval mode (dev set, tuning evidence) | hit@5 | MRR@10 |
|---|---|---|
| Dense embeddings only | 0.93 | 0.940 |
| BM25 only | 0.93 | 0.929 |
| **Hybrid — dense + BM25, RRF fusion (shipped)** | **1.00** | 0.917 |

The two retrievers fail in complementary ways — BM25 cannot find a paraphrase
with zero word overlap; embeddings blur exact names — and fusion covers both.

### What the golden set caught

Freezing a real question set found three things the saturated dev set could
not. All are documented rather than quietly fixed.

**The evaluation set leaked, and the check that finds it is now automated.**
16 of the 68 golden questions were near-duplicates of dev-set questions — 8
word-for-word. The dev set is what chunking, `RERANK_POOL` and the refusal
threshold were tuned against, so those questions measured a system already
optimised for them. Drafting "from the corpus" was not enough: the same corpus
produces the same obvious questions twice. The 16 were replaced with corpus
facts neither set had touched, and v2 measures **0.94 / 0.909 — identical to the
contaminated v1**. All three failures were clean questions in both sets, so the
leakage changed the methodology, not the outcome. It deserved fixing on
principle, not because it flattered the result. The original row is withdrawn
rather than deleted, and `evals/verify_golden.py` now fails any golden question
sharing more than half its words with a dev question. See
[`decisions.md`](decisions.md) 024.

**Rank fusion could not tell silence from a vote — found, then fixed.** A Hindi
question's gold document sat at *dense rank 1* and *hybrid rank 19*, outside the
rerank pool, so the cross-encoder never scored it. BM25 is language-blind and
scores every chunk exactly 0.000 for a Devanagari query — but sorting an
all-zero array still yields an order, and RRF handed that arbitrary order real
weight. Noise outvoted a correct hit.

The fix is one condition: a retriever only votes if it actually discriminated.
It could not be validated at first, because the dev set was saturated at 1.00
and scored identically either way. So the dev set was hardened (21 → 33
questions, weighted to Hindi and thin documents), which separated them cleanly:
Hindi hit@5 0.91 → 1.00, English unchanged. Shipped, and golden v2 re-measured
at **0.96 / 0.925**. See [`decisions.md`](decisions.md) 021 and 025.

**A judge that dies mid-run silently changes the metric.** Faithfulness first
read 0.84, under target. The cause was not the system: the primary judge's
free tier allows 20 requests/day, so it graded the first ~20 answers, hit
quota, and was silently replaced by a fallback — different questions graded by
judges of different strictness. Hand-checking found the "failures" correct and
well grounded. The harness now records which judge graded each answer, keeps
the reason it used to discard, warns when the judge changes mid-run, and
accepts `--judge` to pin one. The number is reported as unmeasured until a
single-judge run completes, rather than quoting a figure known to be invalid.
See [`decisions.md`](decisions.md) 022.

The full phase-by-phase table lives in [`evals/results.md`](evals/results.md).
The methodology — frozen golden set, separate dev set, metric definitions, and
the test suite that checks the answer key itself — is in
[`evals/README.md`](evals/README.md).

## Quickstart

```bash
git clone https://github.com/varshakodi/SchemeSetu.git && cd SchemeSetu
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime deps + tests/eval tooling

python ingest/parse.py        # extract text from any PDFs in data/raw
python naive/rag.py index     # chunk + embed the corpus
python naive/rag.py ask "Who is eligible for PM-KISAN?"
```

Or just `./demo.sh`, which preflights everything (`scripts/doctor.py`) and launches the UI.

Retrieval runs fully offline. Generating grounded, cited answers needs an LLM
provider — **including fully free options** (Groq, Gemini, Cerebras free
tiers, or a local Ollama): set the matching key env var and everything picks
it up automatically via the provider seam in [`agent/llm.py`](agent/llm.py)
(`LLM_PROVIDER` forces a choice; `ANTHROPIC_API_KEY` enables the PRD-default
Claude). Run the eval suite with `python evals/run_eval.py`, and the unit
tests with `pytest`.


Full product requirements: [`PRD.md`](PRD.md)

## Repository layout

```
PRD.md              product requirements — the contract for this build
decisions.md        every technical decision, with alternatives and evidence
naive/              from-scratch implementations: RAG pipeline, BM25
ingest/             corpus loading, cleaning, chunking strategies, PDF parsing
data/registry.csv   provenance for every corpus document (source, date, status)
evals/              golden/dev sets, metric definitions, per-phase results
agent/              LangGraph state machine and the provider-agnostic LLM seam
docs/               interview notes: pitch, definitions, the debugging stories
deploy/             deployment bundle and walkthrough
scripts/doctor.py   preflight check — deps, corpus, models, a live query
tests/              unit tests (run in CI)
```

The corpus itself is not committed — it is rebuildable from the source URLs
in the registry, which also records how complete each captured document is.

## License

[MIT](LICENSE) — the corpus documents are Government of India publications
accessed from public portals; see `data/registry.csv` for per-document
provenance.
