"""SchemeSetu — the chat UI. A thin, deliberately designed shell over the agent.

Design rule (unchanged): the UI adds NO intelligence. It renders the agent's
state — answer, citations, trace — so the evals keep measuring exactly what a
user experiences.

Visual direction: civic-tech. The clarity discipline of public-service design
(strong hierarchy, generous space, high contrast, no decoration for its own
sake) executed with care. Two subject-grounded choices carry it:

  Type — Mukta (Ek Type, an Indian foundry) for UI text and Noto Serif
  Devanagari for display. Both cover Devanagari AND Latin, so a Hindi answer
  and an English one read as one voice instead of two mismatched systems.
  IBM Plex Mono carries identifiers and traces.

  Colour — indigo and burnt saffron: an Indian palette drawn from dye and
  marigold rather than a literal flag, so it reads considered instead of
  clip-art. Semantic colour is separate from the accent: green marks verified
  grounding, amber marks an honest refusal (a refusal is correct behaviour,
  never an error, and must never look like one).

Run locally:  ./demo.sh          Deployed: see deploy/DEPLOY.md
"""

from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent.llm import PROVIDERS, available_provider  # noqa: E402
from naive.rag import CHUNKS_FILE, REFUSAL_THRESHOLD, search  # noqa: E402

st.set_page_config(page_title="SchemeSetu — grounded answers on Indian government schemes",
                   page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")

EXAMPLES = [
    ("Who is not eligible for PM-KISAN?", "eligibility"),
    ("Documents for the SC post-matric scholarship?", "documents"),
    ("मेरी पत्नी पहली बार माँ बनने वाली है, कोई सरकारी मदद?", "हिंदी"),
    ("What subsidy for an electric scooter under FAME-II?", "out of scope"),
]

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Mukta:wght@300;400;500;600;700&family=Noto+Serif+Devanagari:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --paper:#FCFBF8; --panel:#F4F1E9; --card:#FFFFFF;
  --ink:#1A1714; --ink-2:#5F584E; --ink-3:#8A8175;
  --rule:#E3DED2; --indigo:#26346B; --indigo-soft:#EEF0F7;
  --saffron:#B85C10; --verified:#2F6B4F; --amber:#9A6412;
}

/* strip Streamlit chrome so the page reads as a product, not a notebook */
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
#MainMenu, footer, header[data-testid="stHeader"] { display:none !important; }
[data-testid="stAppViewContainer"] { background:var(--paper); }
.block-container { padding-top:2.2rem !important; max-width:1080px; }

html, body, [class*="css"], .stMarkdown, p, li, div, input, textarea, button {
  font-family:'Mukta',-apple-system,system-ui,sans-serif;
  color:var(--ink); font-size:1.02rem; line-height:1.65;
}
h1,h2,h3,h4 { font-family:'Noto Serif Devanagari',Georgia,serif; letter-spacing:-.01em; color:var(--ink); }
code, kbd, .mono { font-family:'IBM Plex Mono',ui-monospace,monospace; font-size:.82rem; }

/* ── masthead ─────────────────────────────────────────────── */
.masthead { display:flex; align-items:baseline; gap:.85rem; flex-wrap:wrap; margin-bottom:.35rem; }
.wordmark { font-family:'Noto Serif Devanagari',serif; font-weight:700; font-size:2.1rem;
            letter-spacing:-.02em; line-height:1.1; }
.wordmark .setu { color:var(--indigo); }
.devanagari { font-family:'Noto Serif Devanagari',serif; color:var(--ink-3); font-size:1.05rem; }
.tagline { color:var(--ink-2); font-size:1.02rem; max-width:64ch; margin:.15rem 0 0; }
.rule { height:3px; margin:1rem 0 1.6rem;
        background:linear-gradient(90deg,var(--indigo) 0 22%,var(--saffron) 22% 34%,var(--rule) 34% 100%); }

/* ── sidebar as a quiet info column ───────────────────────── */
[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--rule); }
[data-testid="stSidebar"] .block-container { padding-top:2rem; }
.side-h { font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:.12em;
          text-transform:uppercase; color:var(--ink-3); margin:1.5rem 0 .5rem; }
.stat { display:flex; justify-content:space-between; align-items:baseline;
        padding:.42rem 0; border-bottom:1px solid var(--rule); }
.stat .k { color:var(--ink-2); font-size:.92rem; }
.stat .v { font-family:'IBM Plex Mono',monospace; font-weight:500; font-variant-numeric:tabular-nums; }
.stat .v.good { color:var(--verified); }
.pill { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:.7rem;
        padding:.18rem .5rem; border-radius:2px; background:var(--indigo-soft);
        color:var(--indigo); border:1px solid #DCE1EF; }
.pill.warn { background:#FBF3E4; color:var(--amber); border-color:#EFDFC0; }
.side-note { color:var(--ink-3); font-size:.84rem; line-height:1.5; }
.side-note a, .stMarkdown a { color:var(--indigo); text-decoration:underline; text-underline-offset:2px; }

/* ── chat ─────────────────────────────────────────────────── */
[data-testid="stChatMessage"] { background:transparent; padding:.2rem 0 1rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  border-left:3px solid var(--rule); padding-left:1rem; margin:1.2rem 0 .6rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  border-left:3px solid var(--indigo); padding-left:1rem; background:var(--card);
  border-radius:0 4px 4px 0; box-shadow:0 1px 2px rgba(26,23,20,.05); padding-top:1rem; }

/* ── example prompts ──────────────────────────────────────── */
.stButton>button { background:var(--card); border:1px solid var(--rule); border-radius:2px;
  color:var(--ink); font-size:.9rem; font-weight:400; text-align:left; padding:.6rem .8rem;
  line-height:1.35; height:100%; transition:border-color .15s, box-shadow .15s; }
.stButton>button:hover { border-color:var(--indigo); box-shadow:0 1px 3px rgba(38,52,107,.14);
  color:var(--indigo); }
/* Streamlit centres button labels on a generated inner div — override on a
   stable selector rather than the hashed emotion class, which changes builds */
.stButton button div, .stButton button p { text-align:left !important; }

/* the chat input auto-sizes to ~270px when rendered inside a tab; hold it to
   one comfortable line that still grows for long questions */
[data-testid="stChatInputTextArea"] { min-height:2.75rem !important; max-height:8rem !important; }
[data-testid="stChatInput"] { border:1px solid var(--rule); border-radius:3px; background:var(--card); }
[data-testid="stChatInput"]:focus-within { border-color:var(--indigo); }

/* ── citations ────────────────────────────────────────────── */
.src { border:1px solid var(--rule); border-left:2px solid var(--indigo); background:var(--card);
       padding:.7rem .85rem; margin-bottom:.6rem; border-radius:0 3px 3px 0; }
.src-head { display:flex; align-items:center; gap:.6rem; margin-bottom:.3rem; flex-wrap:wrap; }
.src-n { font-family:'IBM Plex Mono',monospace; font-weight:500; color:var(--indigo); font-size:.8rem; }
.src-id { font-family:'IBM Plex Mono',monospace; font-size:.76rem; color:var(--ink-3); }
.meter { flex:1; min-width:70px; max-width:130px; height:4px; background:var(--rule); border-radius:2px; }
.meter i { display:block; height:100%; background:var(--verified); border-radius:2px; }
.src-q { color:var(--ink-2); font-size:.9rem; line-height:1.55; margin:0; }

/* ── trace + refusal ──────────────────────────────────────── */
.trace { font-family:'IBM Plex Mono',monospace; font-size:.78rem; color:var(--ink-2);
         background:var(--panel); border:1px solid var(--rule); padding:.7rem .85rem;
         border-radius:3px; word-break:break-word; }
.trace b { color:var(--indigo); font-weight:500; }
.refusal { border-left:3px solid var(--amber); background:#FDF9F0; padding:.9rem 1.1rem;
           border-radius:0 4px 4px 0; }
.refusal .lbl { font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.1em;
                text-transform:uppercase; color:var(--amber); display:block; margin-bottom:.3rem; }
.meta { color:var(--ink-3); font-size:.8rem; font-family:'IBM Plex Mono',monospace; }

[data-testid="stExpander"] { border:none !important; box-shadow:none !important; }
[data-testid="stExpander"] summary { font-size:.85rem; color:var(--ink-2); font-weight:500; }
[data-testid="stTabs"] button p { font-size:.95rem; font-weight:500; }
[data-testid="stChatInput"] textarea { font-size:1rem; }
</style>
"""


@st.cache_resource(show_spinner=False)
def get_agent():
    from agent.graph import build_graph
    return build_graph()


@st.cache_data
def corpus_stats() -> tuple[int, int]:
    records = json.loads(CHUNKS_FILE.read_text())
    return len(records), len({r["doc_id"] for r in records})


def render_result(state: dict, elapsed: float) -> None:
    """Answer, then the evidence behind it. Refusals get their own treatment —
    an honest 'I don't know' is a correct outcome, not an error state."""
    response = state.get("response", "")
    body = response.split("Sources:")[0].strip()
    answered = "Sources:" in response

    if answered:
        st.markdown(body)
    else:
        st.markdown(f'<div class="refusal"><span class="lbl">No grounded answer</span>'
                    f'{html.escape(body)}</div>', unsafe_allow_html=True)

    if answered and state.get("hits"):
        with st.expander(f"Sources — {len(state['hits'])} passages from the corpus"):
            for i, h in enumerate(state["hits"], 1):
                score = h.get("rerank")
                meter = (f'<span class="meter"><i style="width:{max(0, min(1, score)) * 100:.0f}%"></i></span>'
                         f'<span class="src-id">{score:.2f}</span>') if score is not None else ""
                st.markdown(
                    f'<div class="src"><div class="src-head"><span class="src-n">[{i}]</span>'
                    f'<span class="src-id">{html.escape(h["chunk_id"])}</span>{meter}</div>'
                    f'<p class="src-q">{html.escape(" ".join(h["text"].split())[:340])}…</p></div>',
                    unsafe_allow_html=True)

    with st.expander("Trace — the agent's path through the graph"):
        path = "  <b>→</b>  ".join(html.escape(p) for p in state.get("path", []))
        st.markdown(f'<div class="trace">{path}</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="meta">Evidence threshold {REFUSAL_THRESHOLD} · below it the agent '
                    f'rewrites the query, then refuses rather than guessing · answered in '
                    f'{elapsed:.1f}s</p>', unsafe_allow_html=True)


def run_question(question: str) -> None:
    st.session_state.history.append({"role": "user", "text": question})
    with st.chat_message("assistant"):
        start = time.perf_counter()
        if available_provider():
            with st.spinner("Retrieving, grading evidence, grounding the answer…"):
                state = get_agent().invoke({"question": question, "path": []})
        else:
            state = {"response": "No language model is configured on this deployment, so "
                                 "only retrieval is available. The passages below are what "
                                 "the system found.\n\nSources:",
                     "hits": search(question), "path": ["retrieve (no LLM configured)"]}
        elapsed = time.perf_counter() - start
        render_result(state, elapsed)
    st.session_state.history.append({"role": "assistant", "state": state, "elapsed": elapsed})


st.markdown(STYLE, unsafe_allow_html=True)

# ── sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="wordmark">Scheme<span class="setu">Setu</span></div>'
                '<p class="side-note" style="margin-top:.2rem">Bridging citizens and '
                'the schemes they qualify for.</p>', unsafe_allow_html=True)

    n_chunks, n_docs = corpus_stats()
    st.markdown('<div class="side-h">Corpus</div>'
                f'<div class="stat"><span class="k">Documents</span><span class="v">{n_docs}</span></div>'
                f'<div class="stat"><span class="k">Indexed passages</span><span class="v">{n_chunks}</span></div>'
                '<div class="stat"><span class="k">Languages</span><span class="v">EN · हिं</span></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="side-h">Measured quality</div>'
                '<div class="stat"><span class="k">Refuses out-of-corpus</span><span class="v good">1.00</span></div>'
                '<div class="stat"><span class="k">Wrongly refuses</span><span class="v good">0.00</span></div>'
                '<div class="stat"><span class="k">Faithfulness</span><span class="v good">1.00</span></div>'
                '<div class="stat"><span class="k">Hindi ⇄ English parity</span><span class="v good">exact</span></div>'
                '<p class="side-note" style="margin-top:.5rem">Judged by an independent model. '
                'Full method and per-phase history in the eval report.</p>', unsafe_allow_html=True)

    provider = available_provider()
    st.markdown('<div class="side-h">Runtime</div>' + (
        f'<span class="pill">{provider} · {PROVIDERS.get(provider, {}).get("model", "?")}</span>'
        if provider else '<span class="pill warn">retrieval only — no model configured</span>'),
        unsafe_allow_html=True)

    st.markdown('<div class="side-h">Project</div>'
                '<p class="side-note">'
                '<a href="https://github.com/varshakodi/SchemeSetu">Source code</a> · '
                '<a href="https://github.com/varshakodi/SchemeSetu/blob/main/evals/results.md">Eval report</a> · '
                '<a href="https://github.com/varshakodi/SchemeSetu/blob/main/decisions.md">Decision log</a>'
                '</p><p class="side-note">Answers are grounded in official scheme documents '
                '(provenance in <span class="mono">data/registry.csv</span>) and are not '
                'professional advice — verify against the source before acting.</p>',
                unsafe_allow_html=True)

# ── masthead ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="masthead"><span class="wordmark">Scheme<span class="setu">Setu</span></span>'
    '<span class="devanagari">स्कीम सेतु</span></div>'
    '<p class="tagline">Ask about any Indian government scheme in English or हिंदी. '
    'Every answer is grounded in official documents and cited to the passage it came '
    'from — and when the corpus does not hold the answer, it says so.</p>'
    '<div class="rule"></div>', unsafe_allow_html=True)

tab_chat, tab_evals, tab_about = st.tabs(["Ask", "Eval report", "About"])

with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []

    clicked = None
    if not st.session_state.history:
        st.markdown('<div class="side-h" style="margin-top:0">Try one</div>', unsafe_allow_html=True)
        for col, (example, kind) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
            if col.button(example, key=f"ex_{kind}", use_container_width=True,
                          help=f"Example: {kind}"):
                clicked = example

    for turn in st.session_state.history:
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.markdown(turn["text"])
        else:
            with st.chat_message("assistant"):
                render_result(turn["state"], turn["elapsed"])

    question = st.chat_input("Ask about a scheme — eligibility, benefits, documents…")
    question = question or clicked
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        run_question(question)
        st.rerun()

with tab_evals:
    st.markdown((ROOT / "evals" / "results.md").read_text())

with tab_about:
    st.markdown((ROOT / "README.md").read_text().split("## Quickstart")[0])
