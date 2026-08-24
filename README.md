# SchemeSetu

Bilingual (English/Hindi) RAG assistant for Indian government schemes — grounded
answers with per-claim citations, honest refusal when the corpus doesn't know,
and published eval numbers. Built as an 8-week learning project; the full spec
lives in [PRD.md](PRD.md).

**Status: Week 1** — naive RAG from scratch (no frameworks), sample corpus,
golden-set scaffolding.

## Why this exists, and what's different

Tools in this space exist — myScheme's browse/filter portal, Yojana AI, and
per-scheme government chatbots (PM-KISAN's assistant, Ayushman Sarathi).
SchemeSetu differs in three deliberate ways:

1. **Per-claim citations** to the exact source passage, so answers are verifiable.
2. **Measured honest refusal** — out-of-corpus questions get "I don't know",
   and the refusal rate is a tracked metric, because a wrong eligibility answer
   costs a citizen real time and money.
3. **A public eval report** — a frozen 55-question golden set, re-run after
   every build phase, with before/after numbers in [evals/results.md](evals/results.md).

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python naive/rag.py index                                   # build the index
python naive/rag.py ask "Who is not eligible for PM-KISAN?" # retrieve
export ANTHROPIC_API_KEY=sk-ant-...                         # optional: enables answers
python naive/rag.py ask "Who is not eligible for PM-KISAN?" # retrieve + grounded answer
```

Retrieval runs fully offline; only answer generation needs an API key.

## Layout

```
PRD.md            product requirements — the contract for this build
decisions.md      decision log (why each technical choice was made)
naive/rag.py      Week 1: the whole RAG pipeline, from scratch, heavily commented
data/samples/     placeholder scheme documents (replaced by the real corpus in Week 1)
data/raw/         the real corpus (gitignored; see data registry once built)
evals/            golden set, dev set, metric definitions, results table
```

## Eval philosophy

The golden set is written from the documents **before** retrieval is tuned,
then frozen — tuning happens on a separate dev set. Every phase of the build
appends one row of metrics to `evals/results.md`. If a number isn't in that
table, the improvement didn't happen.
