# Evals — how SchemeSetu measures itself

**The core rule: write the questions BEFORE building retrieval, then freeze them.**
If you write test questions after seeing what your system retrieves well, you
unconsciously write questions it can pass. That bias is called **data leakage**,
and it makes your numbers fiction. (Interview keyword — know it.)

## Files

- `golden_set.jsonl` — the frozen exam. Never used for tuning.
- `golden_candidates.jsonl` — the draft awaiting review; becomes the golden set
  when frozen (see [Freezing](#freezing)).
- `dev_set.jsonl` — 21 questions for day-to-day tuning. Tune here, never on the
  golden set.
- `results.md` — one row of metrics per build phase. This table is the interview story.
- `verify_golden.py` — tests the answer key itself.
- `review_golden.py` — prints each question beside the passage that supports it.

## Golden set vs dev set

| | dev set | golden set |
|---|---|---|
| purpose | tuning — chunk size, pool size, thresholds | final reporting |
| run | constantly, during development | once, after freezing |
| may I change the system after seeing the results? | yes, that is the point | **no** |

Every time you look at a result and adjust a parameter, you fit the system a
little more tightly to those particular questions. Numbers from a set you tuned
on describe how well the system does on questions you already optimised for,
which is not a claim anyone should believe.

That is why `RERANK_POOL` was re-swept on the dev set when the corpus grew to
23 documents (decisions 019), and not here.

## Record format (one JSON object per line — "JSONL")

```json
{"id": "g-001", "type": "factual", "language": "en",
 "question": "...", "answer": "ground-truth answer written from the documents",
 "gold_doc_ids": ["pm_kisan.md"]}
```

`type`: factual | synthesis | hindi | trap | ambiguous.
`gold_doc_ids`: which document(s) contain the answer — empty for traps and
ambiguous questions.

For traps and ambiguous questions the `answer` field holds a **pass criterion**,
not a string to match, because correctness there is about behaviour rather than
content. (Earlier drafts used the bare token `REFUSE`; that could not express
the distinction that matters — see Grading.)

## Composition

| slice | count | what it measures |
|---|---|---|
| factual | 33 | can it find one specific fact and cite it |
| synthesis | 10 | can it combine several documents |
| hindi | 10 | cross-lingual retrieval against an English corpus |
| trap | 10 | does it decline what it does not have |
| ambiguous | 5 | does it avoid confident answers to underspecified questions |
| **total** | **68** | PRD §9 asks for 55 |

The trap and ambiguous slices matter most. A benefits assistant that invents a
plausible answer sends a real person to a government office for a scheme they
cannot get.

## Metrics (defined once, measured every phase)

- **hit@5** — fraction of questions where a gold document appears in the top-5
  retrieved chunks. Pure retrieval quality; no LLM involved.
- **MRR@10** — 1/rank of the first correct source, averaged. Rewards ranking
  the right chunk *first*, not just somewhere in the top 10.
- **Faithfulness** — are the answer's claims supported by the retrieved context?
  (LLM-as-judge + 20% checked by hand.)
- **Trap refusal rate** — fraction of trap questions correctly refused.
- **False refusal rate** — fraction of answerable questions wrongly refused.

## Grading

Factual, synthesis and Hindi questions are graded by an independent LLM judge
(`run_agent_eval.py`) on whether the answer is faithful to the retrieved context
and cites the gold document.

Traps and ambiguous questions are graded against the criterion in their `answer`
field:

- **trap** — passes if it declines and says the topic is not in its documents.
  Answering correctly *from general knowledge* is a failure: the citation would
  be unbacked, and the next such answer will be wrong with equal confidence.
- **ambiguous** — passes if it asks for the missing detail, or states the
  assumption it is answering under. The agent has no clarify node (its
  classifier knows only chitchat / in_domain / out_of_domain), so this slice
  measures a known gap rather than a built feature. Recorded honestly: whatever
  it scores is the real number.

## Testing the answer key

The eval set is data that decides whether the system passes. If the data is
wrong, every number downstream is wrong and looks fine. So the answer key has
its own tests:

```bash
python evals/verify_golden.py     # structure, gold docs, figures, traps, composition
python evals/review_golden.py     # each question beside its supporting passage
```

`verify_golden.py` checks that every cited document exists, that **every figure
in an answer appears in the documents it cites**, that every trap subject is
genuinely absent from the corpus, and that slice counts meet the PRD.

The figure check earns its keep. The first draft of g-030 said beneficiaries are
identified from "SECC 2011" data — true in the world, but the corpus says only
"SECC". An answer key containing facts the system was never given is unfair
grading: it cannot cite what it does not have.

The trap check is the one that rots. A trap is only a trap while the corpus
cannot answer it. Ingest a Mudra document and g-051 silently stops testing
refusal and starts testing retrieval, with no error anywhere. **Re-run
`verify_golden.py` after every corpus change.**

## Freezing

1. Review the candidates against the sources — `python evals/review_golden.py`.
2. Edit, cut, replace. Questions you wrote yourself are worth more than drafted
   ones, particularly the traps.
3. `cp evals/golden_candidates.jsonl evals/golden_set.jsonl`
4. Record the freeze date at the top of `results.md`.
5. Run once. Report whatever comes back, including the parts that fail.

After step 5, changing the system means the reported numbers no longer describe
it. If a change is needed, tune on the dev set and say plainly in `results.md`
that the golden numbers predate it. A revised set would be v2 in a new file,
with old results kept and marked v1.

## Process

1. Write golden set v1 from the real corpus documents. Freeze it — record the
   date in `results.md` and never edit it again.
2. End of every phase: run the full eval, append one row to `results.md`.
3. Tune chunk sizes, retrieval and prompts using `dev_set.jsonl` only.
4. Strategy experiments (e.g. `compare_chunkers.py`) build throwaway in-memory
   indexes so the on-disk index and recorded results stay untouched until a
   winner is chosen.
