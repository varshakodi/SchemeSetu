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
