# Evals — how SchemeSetu measures itself

**The core rule: write the questions BEFORE building retrieval, then freeze them.**
If you write test questions after seeing what your system retrieves well, you
unconsciously write questions it can pass. That bias is called **data leakage**,
and it makes your numbers fiction. (Interview keyword — know it.)

## Files

- `golden_set.jsonl` — the frozen exam. 55 questions, never used for tuning.
  Composition target (PRD §9):
  - 20 single-document factual (eligibility, amounts, deadlines, documents)
  - 10 multi-document synthesis (comparisons across schemes)
  - 10 Hindi (mix of the above)
  - 10 out-of-corpus traps (plausible, but unanswerable from our documents)
  - 5 ambiguous (correct behaviour = ask a clarifying question / state assumptions)
- `dev_set.jsonl` — ~10 questions for day-to-day tuning. Tune here, never on the golden set.
- `results.md` — one row of metrics per build phase. This table is the interview story.

## Record format (one JSON object per line — "JSONL")

```json
{"id": "gs-001", "type": "factual", "language": "en",
 "question": "...", "answer": "ground-truth answer written from the documents",
 "gold_doc_ids": ["pm_kisan.md"]}
```

`type`: factual | synthesis | hindi | trap | ambiguous.
`gold_doc_ids`: which document(s) contain the answer — empty for traps.
For traps, set `answer` to `"REFUSE"` — the correct behaviour is to say "I don't know".

## Metrics (defined once, measured every phase)

- **hit@5** — fraction of questions where a gold document appears in the top-5
  retrieved chunks. Pure retrieval quality; no LLM involved.
- **MRR@10** — 1/rank of the first correct source, averaged. Rewards ranking
  the right chunk *first*, not just somewhere in the top 10.
- **Faithfulness** — are the answer's claims supported by the retrieved context?
  (LLM-as-judge + 20% checked by hand.)
- **Trap refusal rate** — fraction of trap questions correctly refused.
- **False refusal rate** — fraction of answerable questions wrongly refused.

## Process

1. Week 1: write golden set v1 from the real corpus documents. Freeze it —
   record the freeze date in `results.md` and never edit it again (v2 would be
   a new file, with old results marked as v1).
2. End of every phase: run the full eval, append one row to `results.md`.
3. Tune chunk sizes, retrieval, prompts using `dev_set.jsonl` only.
4. Strategy experiments (e.g. `compare_chunkers.py`) build throwaway in-memory
   indexes so the on-disk index and recorded results stay untouched until a
   winner is chosen.
