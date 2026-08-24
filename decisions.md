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
