# SchemeSetu — interview notes

Written for future-you, months after the last commit. Everything here is
traceable to `decisions.md`, `evals/results.md` or the code. Nothing is
rounded up.

---

## The 30-second answer to "tell me about your project"

> SchemeSetu is a bilingual question-answering system over Indian government
> welfare scheme documents. You ask in English or Hindi — "am I eligible for
> PM-KISAN?" — and it answers from 23 official documents with a citation to
> the exact passage, or it says it doesn't know. The "doesn't know" part is
> the hard part: a citizen acting on a hallucinated eligibility rule wastes
> weeks and real money. So I built it evaluation-first — a frozen set of 68
> questions, ten of which are deliberately unanswerable, and every
> architectural choice is justified by measured before-and-after numbers.

If they want more, the next sentence is: *"Every stage is written from scratch
first — BM25 in about sixty lines, vector search in plain NumPy — and
frameworks only enter where they earn their keep."*

---

## Resume bullets

Use the numbers exactly as written. Every one is defensible from the repo.

- Built a bilingual (English/Hindi) RAG system over 23 official Indian
  government scheme documents; **hit@5 0.94, MRR@10 0.909** on a frozen
  68-question evaluation set held out from all tuning.
- Implemented hybrid retrieval from scratch — BM25 (~60 lines), dense vector
  search in NumPy, Reciprocal Rank Fusion — plus cross-encoder reranking;
  measured each stage's contribution independently.
- Achieved **100% refusal on out-of-corpus questions** (10/10) at an **8%
  false-refusal rate**, treating honest "I don't know" as a first-class,
  measured requirement rather than a fallback.
- Achieved **exact Hindi/English retrieval parity** by replacing an
  English-only embedder with BGE-M3 cross-lingual embeddings, chosen over
  translate-then-retrieve by A/B measurement (1.000 vs 0.875 MRR).
- Orchestrated the agent as a LangGraph state machine (classify → retrieve →
  rewrite-and-retry → generate → verify → refuse) with layered refusal;
  agent-level evaluation caught a failure mode component metrics could not see.
- Deployed the full-fidelity stack within a **1 GB free-tier memory limit**
  (906 MB measured steady-state) at **₹0 running cost**, using a
  provider-agnostic LLM seam across free API tiers.

**Do not claim a faithfulness number.** It is not yet validly measured — see
below. Claiming it is the one thing that could unravel the interview.

---

## Jargon you must be able to define cold

| Term | One-sentence answer |
|---|---|
| **RAG** | Retrieve relevant documents first, then have the model answer *from* them, so claims are grounded in sources instead of memory. |
| **Chunking** | Splitting documents into passages small enough to retrieve precisely; I used structure-aware chunking on headings, which beat fixed-size windows. |
| **Embedding** | A vector that positions text by meaning, so similar meanings sit close together even with no shared words. |
| **Cosine similarity** | The angle between two vectors — how aligned their meanings are, independent of length. |
| **BM25** | Keyword ranking: term frequency, inverse document frequency, with saturation so a word repeated 50 times doesn't score 50×, and length normalisation. |
| **Hybrid retrieval** | Running keyword and vector search together because they fail differently — BM25 misses paraphrases, embeddings blur exact names. |
| **RRF** | Merging two ranked lists by summing 1/(60+rank) from each. You fuse *ranks*, not scores, because cosine (−1..1) and BM25 (unbounded) aren't comparable. |
| **Bi-encoder vs cross-encoder** | A bi-encoder embeds query and document separately (fast, precomputable); a cross-encoder reads both together (slow, far more accurate). |
| **Two-stage retrieval** | Bi-encoder nominates a cheap candidate pool for recall; cross-encoder re-scores it for precision. |
| **Data leakage** | Letting test questions influence the system, e.g. by tuning until they pass — the numbers then describe questions you optimised for, not real performance. |
| **LLM-as-judge** | Using a model to grade answers. Only meaningful if the judge is a *different model* from the generator — otherwise it's self-assessment. |
| **LangGraph** | A framework for agents as state machines: nodes do work, conditional edges route, and cycles allow retry loops. |

---

## The three stories worth telling

Interviewers remember debugging stories, not architecture diagrams. Each of
these shows a different kind of thinking.

### 1. The measurement that reversed a decision *(judgement)*

I measured the deployed memory footprint at 1364 MB against a 1024 MB free-tier
cap and concluded no free host could run it, and proposed a reduced
configuration. That was wrong twice: I'd used peak memory (`ru_maxrss`,
inflated by model-download buffers) instead of steady-state, in a process that
had already loaded other models. Re-measured cleanly, one process per
configuration: **906 MB**. It fit. Same models, same quality, nothing to
re-tune.

**The lesson:** peak and steady-state answer different questions, and an
inconvenient measurement deserves the same scrutiny as a convenient one. This
one nearly cost a working deployment. *(decisions 018)*

### 2. Rank fusion cannot tell silence from a vote *(depth)*

A Hindi question missed entirely. Its gold document was at **dense rank 1** and
**hybrid rank 19** — outside the rerank pool, so the cross-encoder never saw
it and correctly scored everything it *did* see at zero.

BM25 is language-blind: for a Devanagari query it scores all 132 chunks at
exactly 0.000. But sorting an all-zero array still returns an order — an
arbitrary tie order — and RRF hands the first 50 of those a real 1/(60+rank)
vote. A chunk that dense ranked 5th and BM25 "ranked" 3rd by accident beat the
chunk dense ranked 1st. **Noise outvoted the right answer.**

**The lesson:** rank fusion assumes every input is an opinion. A retriever that
scored nothing has no opinion, and passing its tie order into RRF launders
noise into evidence. *(decisions 021)*

If asked *"why didn't you fix it?"* — the fix is written. It isn't applied
because the dev set is saturated at 1.00 and cannot validate it, and changing
retrieval after freezing the evaluation set would invalidate the reported
numbers. **That answer is stronger than a fix.**

### 3. A judge that died mid-run *(rigour)*

Faithfulness came back 0.84 against a 0.90 target. Before touching the system,
I hand-checked the failures — the answers were correct and well grounded.

The cause was the instrument. The primary judge's free tier allows 20 requests
per day; it graded roughly the first 20 answers, hit quota, and my fallback
chain silently promoted a second judge. Different questions were graded by
judges of different strictness. **Resilience for a demo is corruption for a
measurement.**

**The lesson:** a metric is only as trustworthy as the instrument, and
"graceful degradation" in a measurement path is a bug. The harness now records
which judge graded each answer, keeps the reason it used to discard, warns when
the judge changes mid-run, and can pin one. *(decisions 022)*

---

## Questions they will actually ask

**"How do you know it isn't hallucinating?"**
Three layers, each measured. A retrieval confidence threshold rejects topically
distant questions. Generation replies `NO_ANSWER` when passages are close but
lack the answer — this catches near-misses a threshold can't, like a PMAY-Urban
question retrieving PMAY-Gramin. A verifier fails answers about a different
scheme than the one asked. Result: 10/10 on out-of-corpus traps.

**"Why build BM25 yourself instead of using a library?"**
To understand the failure modes. Knowing BM25 scores every chunk zero on a
Devanagari query — and *why* — is what let me diagnose the fusion bug. A
library call would have hidden it.

**"How would you scale this?"**
The brute-force dot product against every chunk vector is one NumPy line; a
vector database like Qdrant is the scale-out version of that line. I deferred
it deliberately — at 132 chunks it would have added a dependency and hidden the
mechanism, with no measurable gain. I'd add it when linear scan becomes the
bottleneck, which is a measurement, not a guess.

**"What's the weakest part?"**
Corpus size. 23 documents against 3,000+ real schemes. Retrieval quality is
measured and good; coverage is the honest limitation. Second is the ambiguous
slice — the classifier has no clarify path, so "Am I eligible?" gets answered
about *some* scheme. It scores 3/5, and I report it rather than hiding it.

**"What would you do differently?"**
Harden the dev set earlier. It saturated at 1.00, and a tuning set with no
discriminating power can't validate anything — which is exactly why the fusion
fix is still unvalidated.

---

## Running the demo months from now

```bash
cd schemeSetu && ./demo.sh
```

`scripts/doctor.py` runs first and checks Python version, dependencies, corpus,
index freshness, cached models, a provider key, plus a live retrieval and a
live LLM call — each failure prints its own fix. If a free tier has died or a
model id has been retired, doctor tells you which, before the demo does.

Demo durability is tiered on purpose: **recorded video → local run → hosted
link**. A hosted link can die on a vendor's schedule; the video cannot.
