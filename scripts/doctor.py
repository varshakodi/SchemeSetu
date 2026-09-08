"""SchemeSetu — preflight check. Run this before any demo.

Months from now you will want to show this project to an interviewer, and the
worst possible moment to discover a missing model or an expired key is while
someone is watching. This script checks every prerequisite in order and, for
anything broken, prints the exact command that fixes it.

    python scripts/doctor.py          # check everything
    python scripts/doctor.py --quick  # skip the live LLM call
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, BAD = "\033[32m✓\033[0m", "\033[31m✗\033[0m"
failures: list[str] = []


def check(label: str, passed: bool, detail: str = "", fix: str = "") -> bool:
    print(f"  {OK if passed else BAD} {label}" + (f"  — {detail}" if detail else ""))
    if not passed and fix:
        print(f"      fix: {fix}")
        failures.append(label)
    return passed


def main() -> None:
    quick = "--quick" in sys.argv
    print("\nSchemeSetu preflight\n" + "-" * 60)

    # 1. Dependencies — the venv must be the one running this file.
    print("Environment")
    check("python 3.11+", sys.version_info >= (3, 11), sys.version.split()[0],
          "install python 3.11 or newer")
    for mod, why in [("numpy", "vector math"), ("sentence_transformers", "embeddings"),
                     ("langgraph", "the agent graph"), ("streamlit", "the UI"),
                     ("httpx", "LLM providers")]:
        try:
            importlib.import_module(mod)
            check(f"{mod} ({why})", True)
        except ImportError:
            check(f"{mod} ({why})", False, "not installed",
                  ".venv/bin/pip install -r requirements.txt")

    # 2. Corpus and index — retrieval is impossible without them.
    print("\nCorpus and index")
    docs = list((ROOT / "data" / "raw").glob("*.md")) + \
           list((ROOT / "data" / "raw").glob("*.txt")) + \
           list((ROOT / "data" / "samples").glob("*.md"))
    check("corpus documents present", len(docs) > 0, f"{len(docs)} files",
          "see data/registry.csv for every document's source URL")

    index, chunks = ROOT / "naive" / "index.npz", ROOT / "naive" / "chunks.json"
    has_index = index.exists() and chunks.exists()
    check("search index built", has_index,
          f"{index.stat().st_size / 1e6:.1f} MB" if has_index else "missing",
          ".venv/bin/python naive/rag.py index")
    if has_index and docs:
        stale = max(d.stat().st_mtime for d in docs) > index.stat().st_mtime
        check("index newer than corpus", not stale,
              "corpus changed since indexing" if stale else "up to date",
              ".venv/bin/python naive/rag.py index")

    # 3. Models — cached locally, so a demo needs no download and no fast wifi.
    print("\nModels (cached on disk — no download needed at demo time)")
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    for repo, role in [("models--BAAI--bge-m3", "multilingual embedder"),
                       ("models--BAAI--bge-reranker-v2-m3", "multilingual reranker")]:
        path = hub / repo
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9 if path.exists() else 0
        check(f"{role}", size > 0.1, f"{size:.1f} GB cached" if size else "not cached",
              "first run downloads it automatically — do this BEFORE a demo, on good wifi")

    # 4. LLM provider — never print the key itself.
    print("\nLLM provider (answers; retrieval works without one)")
    from agent.llm import PROVIDERS, available_provider
    provider = available_provider()
    check("a provider is configured", provider is not None,
          f"{provider} → {PROVIDERS[provider]['model']}" if provider else "none found",
          "get a free key at console.groq.com, then: "
          "echo 'export GROQ_API_KEY=...' >> ~/.zshrc && source ~/.zshrc")

    # 5. End-to-end smoke tests — the only proof that matters.
    print("\nSmoke test")
    if has_index:
        from naive.rag import search
        hits = search("Who is not eligible for PM-KISAN?")
        check("retrieval returns a relevant chunk", bool(hits) and hits[0]["rerank"] > 0.5,
              f"top={hits[0]['doc_id']} score={hits[0]['rerank']:.2f}" if hits else "no hits",
              "rebuild the index")
    if provider and not quick:
        try:
            from agent.llm import get_llm
            reply = get_llm().complete("Reply with exactly: ok", "ping", max_tokens=512)
            check("live LLM call", "ok" in reply.lower(), f"replied {reply[:30]!r}",
                  "check the key is still valid at console.groq.com")
        except Exception as exc:
            check("live LLM call", False, f"{type(exc).__name__}: {exc}"[:90],
                  "key may be expired or the model retired — see decisions.md 013")

    print("-" * 60)
    if failures:
        print(f"\n{len(failures)} problem(s) to fix before demoing: "
              + ", ".join(failures) + "\n")
        sys.exit(1)
    print("\nAll good. Start the demo with:  ./demo.sh\n")


if __name__ == "__main__":
    main()
