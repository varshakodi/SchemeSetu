"""SchemeSetu — the citizen-facing chat UI.

Audience rule: everything on this screen must matter to someone trying to find
out whether they qualify for a government scheme. Engineering evidence — eval
tables, provider names, retrieval scores — belongs in the repository, not in
front of a citizen. The UI still adds no intelligence of its own; it renders
the agent's state, and nothing more.

Visual system: a green rooted in the subject (agriculture, welfare, growth)
kept deep and calm rather than neon, on warm paper, with a gold accent for
emphasis. Colour carries meaning: green marks an answer backed by documents,
amber marks an honest "not found" — never red, because refusing to guess is
correct behaviour, not an error.

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

from agent.llm import available_provider  # noqa: E402
from naive.rag import CHUNKS_FILE, search  # noqa: E402

st.set_page_config(page_title="SchemeSetu — Indian government schemes, explained",
                   page_icon="🌿", layout="centered",
                   initial_sidebar_state="expanded")

SUGGESTIONS = [
    ("Who is eligible for PM-KISAN?", "Farmer income support"),
    ("What documents do I need for the SC post-matric scholarship?", "Student scholarship"),
    ("मेरी पत्नी पहली बार माँ बनने वाली है, कोई सरकारी मदद?", "मातृत्व लाभ"),
    ("How much help does PMAY-G give to build a house?", "Rural housing"),
    ("किसान क्रेडिट कार्ड पर ब्याज दर कितनी है?", "किसान ऋण"),
    ("What health cover does Ayushman Bharat provide?", "Health insurance"),
]

# Plain-language names for the agent's internal steps. Citizens deserve to see
# how an answer was reached — but as reassurance, not as a debug trace.
STEP_WORDS = {
    "classify": "Understood your question",
    "retrieve": "Searched the official documents",
    "rewrite": "Rephrased the search using official wording",
    "generate": "Drafted an answer using only those documents",
    "verify": "Checked every statement against the sources",
    "refuse": "Found nothing that answers this",
    "direct_reply": "Replied directly",
}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Mukta:wght@300;400;500;600;700&family=Noto+Serif+Devanagari:wght@600;700&display=swap');

:root{
  --paper:#FBFBF8; --card:#FFFFFF;
  --green-900:#0E3A2B; --green-700:#17603F; --green:#1C7A51; --green-300:#8FC3A9;
  --green-100:#E4F1EA; --green-50:#F2F8F4;
  --gold:#BE8C2C; --amber:#8A5B12; --amber-bg:#FDF8EE;
  --ink:#15201B; --ink-2:#556059; --ink-3:#87918B; --rule:#E2E8E4;
}

[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],
#MainMenu,footer,header[data-testid="stHeader"]{display:none!important;}
/* Living green ground: three soft light sources that drift slowly against each
   other, so the page breathes instead of sitting flat. Slow and low-contrast on
   purpose — this is a page people read, not a screensaver. Motion is disabled
   for anyone who has asked their system for reduced motion. */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(760px 520px at 12% -6%, rgba(28,122,81,.16), transparent 62%),
    radial-gradient(700px 480px at 88% 2%, rgba(190,140,44,.13), transparent 60%),
    radial-gradient(900px 620px at 50% 108%, rgba(143,195,169,.20), transparent 66%),
    linear-gradient(180deg, #F6FAF7 0%, var(--paper) 42%, #F4F9F5 100%);
  background-attachment: fixed;
  background-size: 200% 200%, 200% 200%, 200% 200%, 100% 100%;
  animation: drift 26s ease-in-out infinite alternate;
}
@keyframes drift{
  0%  {background-position: 8% 0%, 92% 4%, 50% 100%, 0 0;}
  100%{background-position: 0% 6%, 100% 0%, 44% 92%, 0 0;}
}
.hero .mark{animation:rise .5s ease both;}
.hero .sub{animation:rise .5s .06s ease both;}
.hero .bar{animation:rise .5s .12s ease both, glow 4.5s ease-in-out infinite;}
@keyframes glow{0%,100%{opacity:.85; width:76px;} 50%{opacity:1; width:104px;}}
.stButton button{animation:rise .45s ease both;}
@media (prefers-reduced-motion: reduce){
  [data-testid="stAppViewContainer"], .hero .mark, .hero .sub, .hero .bar,
  .stButton button, [data-testid="stChatMessage"]{animation:none!important;}
}
.block-container{padding-top:2.6rem!important; padding-bottom:6rem!important; max-width:820px;}

html,body,[class*="css"],.stMarkdown,p,li,div,input,textarea,button{
  font-family:'Mukta',-apple-system,system-ui,sans-serif; color:var(--ink);
  font-size:1.04rem; line-height:1.68;}
h1,h2,h3{font-family:'Noto Serif Devanagari',Georgia,serif; color:var(--green-900);}

/* ── header ─────────────────────────────────────────── */
.hero{text-align:center; margin-bottom:1.6rem;}
.hero .mark{font-family:'Noto Serif Devanagari',serif; font-size:2.5rem; font-weight:700;
  color:var(--green-900); line-height:1.15; letter-spacing:-.02em;}
.hero .mark span{color:var(--green);}
.hero .sub{color:var(--ink-2); font-size:1.08rem; max-width:52ch; margin:.5rem auto 0;}
.hero .bar{width:76px; height:4px; margin:1.1rem auto 0; border-radius:3px;
  background:linear-gradient(90deg,var(--green) 0%,var(--green-300) 55%,var(--gold) 100%);}

/* ── suggestion grid ────────────────────────────────── */
.sugg-label{text-align:center; color:var(--ink-3); font-size:.9rem; margin:1.8rem 0 .7rem;}
.stButton button{
  width:100%; height:100%; min-height:5.1rem; background:var(--card);
  border:1px solid var(--rule); border-radius:12px; padding:.85rem 1rem;
  color:var(--ink); font-size:.97rem; font-weight:500; line-height:1.45;
  box-shadow:0 1px 2px rgba(21,32,27,.04);
  transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease;}
/* Streamlit centres label text on nested spans/p; the button is not always a
   direct child of .stButton, so match by descendant and cover every child */
.stButton button,.stButton button *{text-align:left!important;}
.stButton button:hover{
  border-color:var(--green-300); transform:translateY(-2px);
  box-shadow:0 6px 18px rgba(28,122,81,.13); color:var(--green-700);}
.stButton button:active{transform:translateY(0);}
.stButton button:focus:not(:active){border-color:var(--green); color:var(--green-700);}

/* ── chat ───────────────────────────────────────────── */
[data-testid="stChatMessage"]{
  background:transparent; padding:.35rem 0 .9rem;
  animation:rise .28s ease both;}
@keyframes rise{from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:none;}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  background:var(--green-50); border:1px solid var(--green-100);
  border-radius:14px; padding:.85rem 1.1rem; margin:.9rem 0 .3rem;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]){
  background:var(--card); border:1px solid var(--rule); border-left:3px solid var(--green);
  border-radius:4px 14px 14px 4px; padding:1.05rem 1.2rem;
  box-shadow:0 2px 10px rgba(21,32,27,.05);}

/* ── sources ────────────────────────────────────────── */
.src{border:1px solid var(--rule); border-left:2px solid var(--green-300);
  background:var(--green-50); padding:.7rem .9rem; margin-bottom:.55rem; border-radius:0 8px 8px 0;}
.src-h{display:flex; align-items:center; gap:.55rem; margin-bottom:.28rem; flex-wrap:wrap;}
.src-n{font-weight:600; color:var(--green-700); font-size:.86rem;}
.src-doc{color:var(--ink-3); font-size:.83rem;}
.src-t{color:var(--ink-2); font-size:.92rem; line-height:1.55; margin:0;}

/* ── steps + not-found ──────────────────────────────── */
.steps{display:flex; flex-wrap:wrap; gap:.4rem;}
.step{background:var(--green-50); border:1px solid var(--green-100); color:var(--green-700);
  border-radius:20px; padding:.24rem .7rem; font-size:.84rem;}
.notfound{background:var(--amber-bg); border:1px solid #EFE1C6; border-left:3px solid var(--gold);
  border-radius:4px 12px 12px 4px; padding:1rem 1.15rem;}
.notfound .h{color:var(--amber); font-weight:600; display:block; margin-bottom:.25rem;}

/* ── sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"]{background:#F7F9F7; border-right:1px solid var(--rule);}
[data-testid="stSidebar"] .block-container{padding-top:2.4rem;}
.sb-mark{font-family:'Noto Serif Devanagari',serif; font-size:1.5rem; font-weight:700;
  color:var(--green-900);}
.sb-mark span{color:var(--green);}
.sb-text{color:var(--ink-2); font-size:.95rem; margin-top:.4rem;}
.sb-data{margin-top:1.5rem; padding:.75rem .9rem; background:var(--green-50);
  border:1px solid var(--green-100); border-radius:10px; color:var(--green-700);
  font-size:.92rem;}
.sb-note{margin-top:2rem; padding-top:1rem; border-top:1px solid var(--rule);
  color:var(--ink-3); font-size:.82rem; line-height:1.55;}

/* ── input ──────────────────────────────────────────── */
[data-testid="stChatInput"]{border:1px solid var(--rule); border-radius:14px;
  background:var(--card); box-shadow:0 3px 14px rgba(21,32,27,.07);}
[data-testid="stChatInput"]:focus-within{border-color:var(--green);
  box-shadow:0 3px 18px rgba(28,122,81,.16);}
[data-testid="stChatInputTextArea"]{min-height:2.9rem!important; font-size:1.02rem;}
[data-testid="stBottomBlockContainer"]{background:transparent; padding-bottom:1.2rem;}
[data-testid="stExpander"]{border:none!important; box-shadow:none!important;}
[data-testid="stExpander"] summary{font-size:.9rem; color:var(--green-700); font-weight:500;}
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


def humanise(path: list[str]) -> list[str]:
    """Turn the agent's internal node names into steps a citizen can read."""
    steps = []
    for node in path:
        name = node.split("(")[0].strip()
        words = STEP_WORDS.get(name)
        if words and (not steps or steps[-1] != words):
            steps.append(words)
    return steps


ACRONYMS = {"pm", "pmay", "pmjay", "pmsby", "pmjjby", "pmmvy", "pmfby", "apy",
            "kcc", "nmmss", "sc", "st", "obc", "faq", "og", "pms", "ab", "g"}


def pretty_doc(doc_id: str) -> str:
    """A filename a citizen can read: 'myscheme_pm_kisan.md' -> 'PM Kisan'."""
    stem = doc_id.rsplit(".", 1)[0].replace("myscheme_", "")
    return " ".join(w.upper() if w.lower() in ACRONYMS else w.capitalize()
                    for w in stem.split("_"))


def render_answer(state: dict) -> None:
    """Three outcomes, three treatments — not two.

    A greeting and a refusal both arrive without a Sources block, so keying off
    that string alone dressed "Hello, I can help with…" in the amber not-found
    box. The agent already records which node produced the reply; read the path
    instead of guessing from the text.
    """
    response = state.get("response", "")
    body = response.split("Sources:")[0].strip()
    path = state.get("path", [])
    refused = any(step.startswith("refuse") for step in path)
    greeted = any(step.startswith("direct_reply") for step in path)
    answered = not refused and not greeted

    if refused:
        st.markdown(f'<div class="notfound"><span class="h">I could not find this in my documents</span>'
                    f'{html.escape(body)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(body)

    if answered and state.get("hits"):
        with st.expander(f"Where this comes from — {len(state['hits'])} official passages"):
            for i, h in enumerate(state["hits"], 1):
                doc = pretty_doc(h["doc_id"])
                st.markdown(
                    f'<div class="src"><div class="src-h"><span class="src-n">[{i}]</span>'
                    f'<span class="src-doc">{html.escape(doc.title())}</span></div>'
                    f'<p class="src-t">{html.escape(" ".join(h["text"].split())[:320])}…</p></div>',
                    unsafe_allow_html=True)

    steps = humanise(state.get("path", []))
    if steps:
        with st.expander("How I found this"):
            st.markdown('<div class="steps">'
                        + "".join(f'<span class="step">{html.escape(s)}</span>' for s in steps)
                        + '</div>', unsafe_allow_html=True)


def answer(question: str) -> None:
    st.session_state.history.append({"role": "user", "text": question})
    with st.chat_message("assistant", avatar="🌿"):
        if available_provider():
            with st.spinner("Reading the official documents…"):
                state = get_agent().invoke({"question": question, "path": []})
        else:
            state = {"response": "I can only search right now, not write answers. "
                                 "Here is what I found in the documents.\n\nSources:",
                     "hits": search(question), "path": ["retrieve"]}
        render_answer(state)
    st.session_state.history.append({"role": "assistant", "state": state})


st.markdown(STYLE, unsafe_allow_html=True)
n_chunks, n_docs = corpus_stats()

# ── sidebar: identity, data source, disclaimer ────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-mark">Scheme<span>Setu</span></div>'
                '<p class="sb-text">Grounded answers about Indian government '
                'schemes in English and Hindi.</p>'
                f'<div class="sb-data">Currently searching across <b>{n_docs} official '
                f'documents</b> — eligibility rules, benefits and required papers.</div>'
                '<p class="sb-note">Answers are generated by AI and are not professional '
                'advice. Always confirm details with the official scheme portal or your '
                'nearest government office before applying.</p>', unsafe_allow_html=True)

# ── main: header, suggestions, conversation ───────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown('<div class="hero"><div class="mark">Scheme<span>Setu</span></div>'
            '<p class="sub">Ask about any government scheme in English or हिंदी — '
            'eligibility, benefits, or the documents you need.</p>'
            '<div class="bar"></div></div>', unsafe_allow_html=True)

picked = None
if not st.session_state.history:
    st.markdown('<p class="sugg-label">Try one of these</p>', unsafe_allow_html=True)
    for row in (SUGGESTIONS[0:2], SUGGESTIONS[2:4], SUGGESTIONS[4:6]):
        for col, (text, topic) in zip(st.columns(2, gap="medium"), row):
            if col.button(text, key=f"s_{topic}", help=topic, use_container_width=True):
                picked = text

for turn in st.session_state.history:
    if turn["role"] == "user":
        with st.chat_message("user", avatar="🙋"):
            st.markdown(turn["text"])
    else:
        with st.chat_message("assistant", avatar="🌿"):
            render_answer(turn["state"])

typed = st.chat_input("Ask about a scheme — eligibility, benefits, documents…")

question = typed or picked
if question:
    with st.chat_message("user", avatar="🙋"):
        st.markdown(question)
    answer(question)
    st.rerun()
