# Deploying SchemeSetu to Hugging Face Spaces (free)

One-time setup, ~10 minutes, no card. The result: a public URL for your resume.

## 1. Create the Space

1. Sign up / log in at huggingface.co (free account).
2. Top-right profile menu → **New Space**.
3. Name: `SchemeSetu` · License: MIT · SDK: **Streamlit** · Hardware:
   **CPU basic (free)** · Public. → **Create Space**.

## 2. Add your API keys as SECRETS (never as files!)

Space page → **Settings** → **Variables and secrets** → **New secret**:

- Name `GROQ_API_KEY`, value = your Groq key → Save.
- Optionally the same for `GEMINI_API_KEY`.

Secrets become environment variables inside the Space — the provider seam
(`agent/llm.py`) picks them up exactly like on your laptop. Without them the
app runs in clearly-labelled retrieval-only mode.

## 3. Build and upload the bundle

On your laptop:

```bash
python deploy/build_space.py
```

Then upload the **contents** of `deploy/space_bundle/` to the Space. Easiest
path (web): Space page → **Files** → **Add file ▾** → **Upload files** →
drag everything inside `space_bundle/` (folders included) → **Commit**.

Git alternative (the Space is a git repo — same ritual you already know):

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/SchemeSetu hf-space
cp -R deploy/space_bundle/. hf-space/ && cd hf-space
git add -A && git commit -m "Deploy SchemeSetu" && git push
```

(The push asks for your HF username and, as password, an **access token** from
huggingface.co/settings/tokens — create one with "Write" permission. Type it
yourself; never share it or commit it.)

## 4. First boot

The Space builds (installs requirements), then the **first query downloads
~4.6 GB of models** — expect several minutes once, after which the Space
stays warm and answers in seconds. Cold starts after long idle repeat the
model download (free-tier Spaces have no persistent disk) — that trade-off
is recorded in decisions.md 017.

Your app will live at: `https://huggingface.co/spaces/YOUR_USERNAME/SchemeSetu`
