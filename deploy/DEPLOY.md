# Deploying SchemeSetu — free, full fidelity

**Target: Streamlit Community Cloud** (free, no card, unlimited public apps).

## Why this works — and why an earlier attempt said it didn't

Hugging Face Spaces removed its free compute tier in 2026 (Static free,
Gradio/Docker PRO-only, Streamlit no longer offered), so that route is closed.
Streamlit Community Cloud caps apps at **1024 MB RAM**, and a first
measurement suggested this project needed 1364 MB — a wrong number, taken
with `ru_maxrss` (*peak* memory, inflated by model-download spikes) in a
process that had already loaded other models.

Measured properly — steady-state RSS, one clean process per configuration,
models already cached:

| stage | RSS |
|---|---|
| app only (numpy + streamlit) | 55 MB |
| + BGE-M3 embedder | 692 MB |
| + bge-reranker-v2-m3 | 869 MB |
| + after serving a real query | **906 MB** |

It fits, at full fidelity — same models, same eval numbers, no reduced
configuration. Headroom is thin (~100 MB once Streamlit's server overhead is
counted), so see *If it runs out of memory* below.

The lesson worth keeping: **peak memory and steady-state memory are different
questions, and mixing models in one process makes both meaningless.** Measure
the thing you actually deploy, in isolation.

## Deploy it

1. **Push everything to GitHub** — the repo is the deployment unit; Streamlit
   Cloud builds directly from it. The search index (`naive/index.npz`,
   `naive/chunks.json`, ~0.6 MB) is committed on purpose so the app never
   rebuilds it and never needs the corpus at runtime.
2. Go to **share.streamlit.io** → sign in with GitHub → **Create app** →
   **Deploy a public app from GitHub**.
3. Repository `varshakodi/SchemeSetu` · Branch `main` · Main file **`app.py`**.
4. **Advanced settings** → Python version **3.11** → **Secrets**, paste
   (TOML format, quotes required — read the value with
   `grep GROQ_API_KEY ~/.zshrc`, never commit it):

   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```

5. **Deploy.** The build installs CPU-only PyTorch (see the comment in
   `requirements.txt` — the default Linux wheel bundles ~2.5 GB of CUDA and
   would blow the build). First query then downloads the models (~4.6 GB,
   several minutes, once). After that it answers in seconds.

Your URL: `https://<something>.streamlit.app` — put it in the README and on
your resume.

## Operating it

- **Sleeping:** the app sleeps after ~12 hours idle and wakes on the next
  visit (slow first load, then normal). Good for placement season: visit it
  the morning of an interview to warm it up.
- **If it runs out of memory** ("this app has gone over its resource limits"):
  the contingency is to serve without the cross-encoder, which frees ~180 MB.
  That is a *different* system — refusal currently keys off reranker scores —
  so it needs a cosine-based threshold re-tuned on the dev set and its own row
  in `evals/results.md` before it goes live. Don't ship it unmeasured.
- **If answers stop working** but search still does: the LLM key or model id
  has drifted (it has happened twice — decisions 013). Run
  `python scripts/doctor.py` locally to confirm, then update the Space secret
  or `agent/llm.py`.

## Keep the local demo too

A hosted link can die on someone else's schedule; a laptop demo cannot.
`./demo.sh` runs the full preflight (`scripts/doctor.py`) and launches the
app locally at full fidelity. And record the 3-minute video — it is the only
demo immune to wifi, quotas, and vendor policy changes. Tiers, most reliable
first: **video → local → hosted**.
