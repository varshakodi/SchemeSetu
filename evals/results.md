# Eval results — SchemeSetu

Golden set: **v1 — FROZEN on: ____ (fill in when frozen)** · dev set used for tuning only.
Targets (PRD §5): hit@5 ≥ 0.85 · faithfulness ≥ 0.90 · trap refusal ≥ 0.90 · false refusal ≤ 0.10.

| Phase | Date | hit@5 | MRR@10 | Faithfulness | Trap refusal | False refusal | p50 latency | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 — naive (fixed chunks, MiniLM, cosine) | | | | | | | | baseline |
| 2 — chunking experiments | | | | | | | | winner: ___ |
| 3 — hybrid (BM25 + dense + RRF) | | | | | | | | |
| 4 — reranker + citations | | | | | | | | |
| 5 — LangGraph agent | | | | | | | | |
| 6 — Hindi (BGE-M3 vs translate) | | | | | | | | winner: ___ |
| 7 — deployed | | | | | | | | measured on public URL |
