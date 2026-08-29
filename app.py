"""SchemeSetu — the chat UI (Week 7). A thin Streamlit shell over the agent.

Design rule: the UI adds NO intelligence. All behaviour lives in the agent
graph; this file only renders the agent's state — answer, citation panel,
trace view — as separate interface elements. Keeping the UI thin means the
evals keep measuring the same system users experience.

Run locally:   streamlit run app.py
Deployed:      Hugging Face Spaces (see deploy/README_SPACE.md)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent.llm import PROVIDERS, available_provider  # noqa: E402
from naive.rag import CHUNKS_FILE, REFUSAL_THRESHOLD, search  # noqa: E402

st.set_page_config(page_title="SchemeSetu", page_icon="🧭", layout="wide")

EXAMPLES = [
    "Who is not eligible for PM-KISAN?",
    "What documents do I need for the SC post-matric scholarship?",
    "मेरी पत्नी पहली बार माँ बनने वाली है, क्या हमें कोई सरकारी मदद मिल सकती है?",
    "पीएम-किसान योजना में हर साल कितने पैसे मिलते हैं?",
]


@st.cache_resource(show_spinner="Loading models (first load takes a while)...")
def get_agent():
    from agent.graph import build_graph
    return build_graph()


@st.cache_data
def corpus_stats() -> tuple[int, int]:
    records = json.loads(CHUNKS_FILE.read_text())
    return len(records), len({r["doc_id"] for r in records})


def render_result(state: dict, elapsed: float) -> None:
    """Answer + citation panel + trace view, reconstructed from agent state."""
    response = state.get("response", "")
    answered = "Sources:" in response
    st.markdown(response.split("Sources:")[0].strip())

    if answered and state.get("hits"):
        with st.expander(f"📚 Sources ({len(state['hits'])} passages)"):
            for i, h in enumerate(state["hits"]):
                score = h.get("rerank")
                score_txt = f" · relevance {score:.2f}" if score is not None else ""
                st.markdown(f"**[{i + 1}] `{h['chunk_id']}`**{score_txt}")
                st.caption(" ".join(h["text"].split())[:400])

    with st.expander("🔍 Trace — the agent's path through the graph"):
        st.code("  →  ".join(state.get("path", [])), language=None)
        st.caption(
            f"Evidence threshold: {REFUSAL_THRESHOLD} (below = refuse rather than "
            f"guess). Answered in {elapsed:.1f}s.")


def run_question(question: str) -> None:
    st.session_state.history.append({"role": "user", "text": question})
    provider = available_provider()
    with st.chat_message("assistant"):
        start = time.perf_counter()
        if provider:
            state = get_agent().invoke({"question": question, "path": []})
        else:  # graceful degradation: retrieval-only, clearly labelled
            hits = search(question)
            state = {"response": "*(No LLM provider configured on this "
                                 "deployment — showing retrieved passages "
                                 "only.)*\n\nSources:", "hits": hits,
                     "path": ["retrieve (no LLM)"]}
        elapsed = time.perf_counter() - start
        render_result(state, elapsed)
    st.session_state.history.append(
        {"role": "assistant", "state": state, "elapsed": elapsed})


# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.title("🧭 SchemeSetu")
    st.markdown("Grounded answers about Indian government schemes — "
                "with citations, or an honest *\"I don't know.\"* "
                "English **and** हिंदी.")

    provider = available_provider()
    if provider:
        st.success(f"LLM provider: **{provider}** "
                   f"(model: {PROVIDERS.get(provider, {}).get('model', '?')})")
    else:
        st.warning("No LLM provider configured — retrieval-only mode. "
                   "Add a GROQ_API_KEY or GEMINI_API_KEY secret to enable answers.")

    n_chunks, n_docs = corpus_stats()
    st.metric("Corpus", f"{n_docs} documents · {n_chunks} chunks")
    st.markdown(
        "**Latest eval** — trap refusal **1.00**, false refusal **0.00**, "
        "faithfulness **1.00** (independent judge) · "
        "Hindi/English retrieval parity: exact.")
    st.markdown("[Code](https://github.com/varshakodi/SchemeSetu) · "
                "[Eval report](https://github.com/varshakodi/SchemeSetu/blob/main/evals/results.md) · "
                "[Decision log](https://github.com/varshakodi/SchemeSetu/blob/main/decisions.md)")
    st.caption("Runs entirely on free tiers. Answers are grounded in the "
               "document corpus (provenance: data/registry.csv) and are not "
               "professional advice — verify with official sources.")

# --- Main area ---------------------------------------------------------------

tab_chat, tab_evals, tab_about = st.tabs(["💬 Chat", "📊 Eval report", "📖 About"])

with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown("**Try one:**")
    cols = st.columns(len(EXAMPLES))
    clicked = None
    for col, example in zip(cols, EXAMPLES):
        if col.button(example[:38] + ("…" if len(example) > 38 else ""),
                      help=example, use_container_width=True):
            clicked = example

    for turn in st.session_state.history:
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.markdown(turn["text"])
        else:
            with st.chat_message("assistant"):
                render_result(turn["state"], turn["elapsed"])

    question = st.chat_input("Ask about any scheme — English or हिंदी")
    if clicked and not question:
        question = clicked
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        run_question(question)
        st.rerun()  # re-render so history (incl. this turn) paints once, cleanly

with tab_evals:
    st.markdown((ROOT / "evals" / "results.md").read_text())
    st.info("Methodology — frozen golden set, separate dev set, metric "
            "definitions: see evals/README.md in the repository.")

with tab_about:
    st.markdown((ROOT / "README.md").read_text().split("## Quickstart")[0])
