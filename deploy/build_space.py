"""Assemble the Hugging Face Space bundle in deploy/space_bundle/.

A Space is just a git repo that HF runs. This script gathers everything the
deployed app needs — code, corpus, PREBUILT index (so the Space never spends
boot time re-embedding), and a Space README with the config frontmatter —
into one folder you upload. The GitHub repo stays the source of truth; this
bundle is a build artifact (deploy/space_bundle/ is gitignored).

Run after any change you want deployed:
    python deploy/build_space.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "deploy" / "space_bundle"

SPACE_README = """\
---
title: SchemeSetu
emoji: 🧭
colorFrom: green
colorTo: yellow
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# SchemeSetu 🧭

Bilingual (English/हिंदी) RAG assistant for Indian government schemes —
grounded answers with citations, honest refusal, and published evals.
Built and evaluated in the open: https://github.com/varshakodi/SchemeSetu

Note: the first query after a cold start downloads the embedding and
reranking models (a few minutes); afterwards the Space stays warm.
"""


def main() -> None:
    if not (ROOT / "naive" / "index.npz").exists():
        sys.exit("No prebuilt index found — run `python naive/rag.py index` first.")

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    for folder in ["naive", "ingest", "agent", "evals", "data"]:
        shutil.copytree(ROOT / folder, BUNDLE / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ["app.py", "requirements.txt", "decisions.md", "LICENSE"]:
        shutil.copy2(ROOT / f, BUNDLE / f)
    (BUNDLE / "README.md").write_text(SPACE_README)

    size_mb = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file()) / 1e6
    print(f"Bundle ready: {BUNDLE}  ({size_mb:.1f} MB)")
    print("Upload its CONTENTS to your Space (see deploy/README_SPACE.md).")


if __name__ == "__main__":
    main()
