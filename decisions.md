# Decision log — SchemeSetu

One entry per meaningful technical decision. This is interview preparation:
panels drill depth-first ("why this chunk size?"), and this file is where the
answers live. Template:

> **NNN — Title (date)**
> **Decision:** what was chosen. **Alternatives:** what else was considered.
> **Why:** the reason — with eval numbers once we have them. **Revisit when:** the trigger to reconsider.

---

**001 — Project renamed YojanaSathi → SchemeSetu (24 Aug 2026)**
**Decision:** SchemeSetu (*setu* = bridge). **Alternatives:** YojanaSathi (original), Patrata, SchemeScout.
**Why:** yojanasaathi.com already exists (a schemes news blog), and the Saathi/Sarathi/Mitra namespace is crowded with government apps. A bilingual name also mirrors a bilingual product. **Revisit when:** never — names are a one-way door once the repo is public.

**002 — Embedding model: all-MiniLM-L6-v2 now, BGE-M3 in Week 6 (24 Aug 2026)**
**Decision:** start with MiniLM (~90 MB, English-only, fast on CPU); the model name is a single config constant in `naive/rag.py`. **Alternatives:** BGE-M3 from day one (~2 GB, multilingual), API embeddings (Voyage/OpenAI — per-call cost).
**Why:** Weeks 1–5 are English-only per the PRD, so multilingual capacity buys nothing yet and the small model keeps the edit-run-learn loop fast. Making the model a config knob means the Week 6 swap is a one-line change we can A/B properly. **Revisit when:** Week 6 (Hindi), or if English retrieval quality plateaus below target.

**003 — Chunking: fixed ~800 chars with 150 overlap (24 Aug 2026)**
**Decision:** paragraph-packed fixed-size chunks as the Week-1 baseline. **Alternatives:** recursive splitting, structure-aware (by headings) — both scheduled for Week 2.
**Why:** simplest possible baseline; its eval numbers are the yardstick the Week-2 strategies must beat. Chosen by simplicity, not by measurement — that's acceptable only for a baseline. **Revisit when:** Week 2, by dev-set numbers.

**004 — Corpus acquisition: browser-rendered myScheme pages + official PDFs (27 Aug 2026)**
**Decision:** extract scheme text from myscheme.gov.in via a real browser (the site is a JavaScript app whose sections load client-side, sometimes flakily), and download guideline PDFs where reachable; every document gets a provenance header and a row in `data/registry.csv`. **Alternatives:** direct portal downloads (pmkisan.gov.in and pmayg.nic.in timed out from this network; agriwelfare.gov.in blocks non-browser clients with 403), the myScheme private API (undocumented, needs a scraped key — rejected), third-party mirrors like InstaPDF/Scribd (weak provenance — last resort only).
**Why:** corpus authenticity and traceability beat convenience; the registry records exactly what was fetched, from where, when, and how complete it is (three pages captured partial sections — recorded, not hidden). Raw corpus files stay out of git (large, redistributable from source); the registry IS committed, so anyone can rebuild the corpus. **Revisit when:** portals become reachable (fetch the canonical guideline PDFs), or corpus needs Hindi documents (Week 6).

**005 — PDF parsing: pypdf (27 Aug 2026)**
**Decision:** `pypdf` for PDF→text, with per-page thin-page warnings. **Alternatives:** PyMuPDF (better layout handling, AGPL license), Docling (best tables/structure, heavy), OCR pipelines (only needed for scanned documents).
**Why:** pure-Python, tiny, sufficient for text-first PDFs like our current guidelines; the parse report tells us when a document needs a heavier tool instead of failing silently. **Revisit when:** a critical document shows thin-page warnings (scanned) or mangled tables — that's the trigger to bring in Docling per PRD §8.

**006 — Chunking: structure-aware with [Title › Section] labels wins (27 Aug 2026)**
**Decision:** default chunker is `structure` (split at markdown headings, keep sections whole, label every chunk with its document title and section; provenance boilerplate stripped by `clean_text` before chunking). **Alternatives raced** (evals/compare_chunkers.py, 4 configs, identical corpus/embedder/questions): fixed-raw (Week-1 replica), fixed-clean, recursive, structure.
**Why — the numbers:** dev MRR@10: fixed 0.929, recursive 0.905, structure **1.000** (only config to fix the NMMSS-vs-PMS-SC scholarship confusion — the label tells the embedding which scheme a chunk belongs to). Mid-word chunk starts: ~65% (fixed/recursive) → **0%** (structure). Trap-vs-answerable top-1 gap: widest under structure (0.33 vs 0.66 on dev). Notable negative result worth remembering: *smarter split boundaries alone (recursive) slightly hurt ranking; the win came from context labels, not prettier cuts.* Cleaning alone changed no dev metric but re-ranked golden questions (see 007). **Revisit when:** corpus gains heading-less document types (label falls back to title only), or Week-4 reranking changes what precision@k needs.

**007 — Gold labels are maintained artifacts, not one-time annotations (27 Aug 2026)**
**Decision:** whenever the corpus grows, re-check (a) every trap — an answer entering the corpus un-traps it (happened with PM-JAY, dev-004), and (b) every `gold_doc_ids` list — a new document that also contains an answer must be added as gold (happened with pm_kisan_faq.txt: cleaning let it outrank the labelled golds, making correct retrieval score as a miss — golden MRR "fell" to 0.750 until the label was fixed).
**Why:** eval numbers are only as true as their answer key; a stale key turns real improvements into phantom regressions and vice versa. This is label maintenance, not tuning — the questions and answers never changed. **Revisit when:** the golden set is frozen — after freezing, label corrections get logged here rather than silently edited.

**008 — BM25 built from scratch; Qdrant deferred to the deploy phase (28 Aug 2026)**
**Decision:** implement BM25 ourselves (`naive/bm25.py`, ~60 lines: TF, smoothed IDF, k1 saturation, length normalisation) instead of the PRD's `rank_bm25` package, and keep retrieval in-process instead of adopting Qdrant this week. **Alternatives:** rank_bm25 (same algorithm, one pip install — the production swap stays trivial), Qdrant now (Docker or cloud signup).
**Why:** the project's learning rule — build the algorithm once by hand before using the library — and this one is small enough to hand-build in an afternoon while being permanent interview material (explain IDF? saturation? we wrote them). Qdrant solves scale/persistence/filtering, none of which binds at 116 chunks; it earns its keep at deployment (Week 7), where its free cloud tier also matters. **Revisit when:** corpus outgrows brute force, metadata filtering (FR-3.3) is needed, or deploy week arrives.

**009 — Retrieval: hybrid (dense + BM25, RRF fusion) is the default (28 Aug 2026)**
**Decision:** `RETRIEVAL_MODE = "hybrid"` — both retrievers run, results fused by Reciprocal Rank Fusion (1/(60+rank), ranks not scores, because cosine and BM25 scales are incomparable). **Alternatives raced** on the hardened 16-question dev v2 (the old dev set was saturated at 1.00 — an exam nothing can fail teaches nothing): dense hit@5 0.93 / MRR 0.940; bm25 0.93 / 0.929; hybrid **1.00** / 0.917.
**Why:** hit@5 is the north-star recall metric — a gold document absent from the top-5 is unrecoverable by anything downstream, while a rank-2/3 result is recoverable. The per-question matrix confirmed complementary failures: BM25 could not find the zero-word-overlap paraphrase ("my wife is expecting..." → maternity benefit, rank >10); dense wandered on the keyword-ish synthesis query ("transfer money directly into bank account", rank 6); hybrid held every question in top-5. Accepted cost: MRR dipped 0.940 → 0.917 (fusion dilutes single-mode wins) — that is deliberately the Week-4 reranker's job, giving the classic two-stage architecture: **hybrid maximises recall into a candidate pool; the reranker restores precision at the top**. Honest surprise worth quoting: dense handled name/abbreviation queries fine *because Week 2's [Title › Section] labels already put names into every chunk* — one improvement pre-empted part of the next. **Revisit when:** the reranker lands (re-tune the pool size), or corpus growth changes the balance.
