# Demoing SchemeSetu — three tiers, all free

**Context (Sept 2026):** Hugging Face Spaces no longer offers a free tier that
runs compute — Static Spaces are free, Gradio/Docker require PRO, and
Streamlit is no longer offered at all. Streamlit Community Cloud is still
free but caps apps at **1 GB RAM**, and this project's models need **1364 MB**
(measured: BGE-M3 alone is 1126 MB). So there is currently no free host that
runs SchemeSetu at full fidelity.

That is a constraint, not a blocker. What matters for placement season is
being able to *show the project reliably months from now* — and a hosted link
that has quietly died is worse than no link at all. Hence three tiers, most
reliable first.

## Tier 1 — The recorded demo (never breaks)

A 3-minute screen recording is the only demo that works with no wifi, no
expired key, and no vendor policy change. Record it once, keep it in the
repo README and on your phone.

macOS: `Cmd+Shift+5` → Record Selected Portion → run through:
one English question (show the Sources panel), one Hindi question (show the
trace with the rewrite loop), one trap (show the honest refusal), then the
Eval report tab. Narrate what each panel proves.

## Tier 2 — The local demo (full fidelity, one command)

```bash
./demo.sh
```

This runs `scripts/doctor.py` first, which verifies python, dependencies,
corpus, index freshness, cached models, and the API key, and prints the exact
fix for anything broken — *before* an interviewer is watching. Run it the day
before, not five minutes before.

**Durability checklist for months-later use**
- The models (~4.6 GB) are cached under `~/.cache/huggingface`. Don't clear
  it. A demo laptop with the cache needs no download and no fast wifi.
- The API key lives in `~/.zshrc`. Keys can expire or be revoked — the doctor
  makes a live call, so you'll know in ten seconds. New free key:
  console.groq.com.
- Provider model ids drift (this project has already been bitten twice —
  decisions 013). If the doctor's live call fails with a 404, list the
  provider's current models and update `agent/llm.py`, or override on the
  spot with `LLM_MODEL=...`.
- Retrieval works with no key at all, so even a dead provider leaves you a
  working demo of search, citations, and the eval report.

## Tier 3 — A hosted URL (optional, best-effort)

To fit a 1 GB free host, the app would run a *reduced* configuration: query
embeddings from a hosted API (Gemini's `gemini-embedding-001` free tier —
verified cross-lingual, 0.78 cosine on a Hindi-question/English-answer pair)
against a precomputed corpus index, keyword BM25 in-process, and no local
cross-encoder — roughly 60 MB of RAM instead of 1364 MB.

That is a genuinely different system, so it would need its own eval run and
its own refusal threshold, and would be published as such: "deployed config"
next to "full config" in `evals/results.md`. Never let a resume link point at
an unmeasured system.

Not built yet — Tier 1 and Tier 2 cover the placement-season need. Build it
only if you want the link, and budget an eval run for it.
