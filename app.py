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
  --bg:#071A13; --bg-2:#0B2418; --surface:#0F2E1F; --surface-2:#143724;
  --line:#1C4531; --line-soft:#163826;
  --ink:#EDF3EE; --ink-2:#A6BCAD; --ink-3:#728B7B;
  --green:#3FBF7F; --green-deep:#1E7A52; --green-glow:rgba(63,191,127,.5);
  --brass:#D9A441; --brass-dim:#8A6A22;
  --warn:#E0A93B; --warn-bg:rgba(224,169,59,.08);
}

[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],
#MainMenu,footer,header[data-testid="stHeader"]{display:none!important;}

/* Aurora ground: two green light sources and one brass, drifting slowly out of
   phase. Slow and low-contrast — this is a page people read. */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(820px 560px at 14% -8%, rgba(63,191,127,.16), transparent 64%),
    radial-gradient(760px 520px at 88% 4%, rgba(217,164,65,.11), transparent 62%),
    radial-gradient(1000px 700px at 52% 112%, rgba(30,122,82,.24), transparent 68%),
    linear-gradient(178deg,#061710 0%, var(--bg) 46%, #05150F 100%);
  background-attachment:fixed;
  background-size:190% 190%,190% 190%,190% 190%,100% 100%;
  animation:aurora 30s ease-in-out infinite alternate;}
@keyframes aurora{
  0%{background-position:10% 0%, 90% 6%, 50% 100%, 0 0;}
  100%{background-position:0% 8%, 100% 0%, 42% 90%, 0 0;}}

/* Film grain. The single most effective defence against a page looking like a
   flat generated gradient — it gives the dark ground a physical texture. */
[data-testid="stAppViewContainer"]::before{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.35;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.42'/%3E%3C/svg%3E");}
.block-container{position:relative; z-index:1; padding-top:2.4rem!important;
  padding-bottom:6rem!important; max-width:840px;}

html,body,[class*="css"],.stMarkdown,p,li,div,input,textarea,button{
  font-family:'Mukta',-apple-system,system-ui,sans-serif; color:var(--ink);
  font-size:1.04rem; line-height:1.7;}
h1,h2,h3{font-family:'Noto Serif Devanagari',Georgia,serif; color:var(--ink);}
strong,b{color:#FFFFFF; font-weight:600;}

/* ── hero: shown only as the empty state ─────────────── */
.hero{text-align:center; margin:1.4rem 0 .4rem;}
.hero .mark{font-family:'Noto Serif Devanagari',serif; font-size:3rem; font-weight:700;
  letter-spacing:-.025em; line-height:1.12; color:var(--ink);
  animation:rise .55s ease both;}
.hero .mark span{
  background:linear-gradient(96deg,var(--green) 0%,#79D9A8 46%,var(--brass) 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
  background-size:220% 100%; animation:sweep 9s ease-in-out infinite;}
@keyframes sweep{0%,100%{background-position:0% 50%;} 50%{background-position:100% 50%;}}
.hero .sub{color:var(--ink-2); font-size:1.1rem; max-width:50ch; margin:.7rem auto 0;
  animation:rise .55s .07s ease both;}
.hero .bar{width:88px; height:3px; margin:1.3rem auto .2rem; border-radius:3px;
  background:linear-gradient(90deg,transparent,var(--green),var(--brass),transparent);
  animation:rise .55s .13s ease both, breathe 5s ease-in-out infinite;}
@keyframes breathe{0%,100%{opacity:.6; width:88px;} 50%{opacity:1; width:132px;}}
@keyframes rise{from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:none;}}

/* ── suggestion cards ────────────────────────────────── */
.sugg-label{text-align:center; color:var(--ink-3); font-size:.86rem;
  letter-spacing:.14em; text-transform:uppercase; margin:2rem 0 .9rem;}
.stButton button{
  position:relative; width:100%; min-height:5.4rem;
  background:linear-gradient(160deg,var(--surface) 0%,var(--bg-2) 100%);
  border:1px solid var(--line-soft); border-radius:14px; padding:.95rem 1.1rem;
  color:var(--ink); font-size:.98rem; font-weight:400; line-height:1.5;
  animation:rise .5s ease both;
  transition:transform .2s cubic-bezier(.2,.7,.3,1), border-color .2s, box-shadow .2s, background .2s;}
/* text-align alone does nothing here: Streamlit's button is a flex
   container, so the label is centred as a flex ITEM. Align the item. */
.stButton button,.stButton button *{text-align:left!important;}
.stButton button{justify-content:flex-start!important; align-items:center!important;}
/* the label sits in a nested flex wrapper that centres it too */
.stButton button > div{justify-content:flex-start!important; width:100%;}
.stButton button:hover{
  transform:translateY(-3px); border-color:var(--green-deep);
  background:linear-gradient(160deg,var(--surface-2) 0%,var(--surface) 100%);
  box-shadow:0 12px 30px rgba(0,0,0,.45), 0 0 0 1px rgba(63,191,127,.18),
             0 0 26px -8px var(--green-glow);}
.stButton button:active{transform:translateY(-1px);}

/* ── chat ────────────────────────────────────────────── */
[data-testid="stChatMessage"]{background:transparent; padding:.3rem 0 .9rem;
  animation:rise .34s ease both;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  background:rgba(63,191,127,.07); border:1px solid var(--line-soft);
  border-radius:16px; padding:.85rem 1.15rem; margin:1.1rem 0 .35rem;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]){
  background:linear-gradient(168deg,var(--surface) 0%,rgba(11,36,24,.86) 100%);
  border:1px solid var(--line); border-left:3px solid var(--green);
  border-radius:4px 16px 16px 4px; padding:1.15rem 1.3rem;
  box-shadow:0 10px 34px rgba(0,0,0,.4);}

/* ── sources ─────────────────────────────────────────── */
.src{background:rgba(7,26,19,.62); border:1px solid var(--line-soft);
  border-left:2px solid var(--green-deep); padding:.75rem .95rem;
  margin-bottom:.6rem; border-radius:0 10px 10px 0; transition:border-color .18s;}
.src:hover{border-left-color:var(--green);}
.src-h{display:flex; align-items:center; gap:.6rem; margin-bottom:.3rem; flex-wrap:wrap;}
.src-n{font-weight:600; color:var(--green); font-size:.87rem;}
.src-doc{color:var(--brass); font-size:.84rem; letter-spacing:.02em;}
.src-t{color:var(--ink-2); font-size:.93rem; line-height:1.6; margin:0;}

/* ── steps + not-found ───────────────────────────────── */
.steps{display:flex; flex-wrap:wrap; gap:.45rem;}
.step{background:rgba(63,191,127,.10); border:1px solid rgba(63,191,127,.24);
  color:#8FD9B2; border-radius:20px; padding:.28rem .8rem; font-size:.85rem;
  animation:rise .4s ease both;}
.notfound{background:var(--warn-bg); border:1px solid rgba(224,169,59,.28);
  border-left:3px solid var(--warn); border-radius:4px 14px 14px 4px; padding:1.05rem 1.2rem;}
.notfound .h{color:var(--warn); font-weight:600; display:block; margin-bottom:.3rem;}

/* ── sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#061912 0%,#082116 100%);
  border-right:1px solid var(--line-soft);}
[data-testid="stSidebar"] .block-container{padding-top:2.6rem;}
.sb-mark{font-family:'Noto Serif Devanagari',serif; font-size:1.6rem; font-weight:700;
  color:var(--ink); line-height:1.2;}
.sb-mark span{color:var(--green);}
.sb-text{color:var(--ink-2); font-size:.95rem; margin-top:.45rem;}
.sb-data{margin-top:1.6rem; padding:.85rem 1rem; border-radius:12px;
  background:linear-gradient(150deg,rgba(63,191,127,.10),rgba(217,164,65,.05));
  border:1px solid var(--line-soft); color:var(--ink-2); font-size:.93rem;}
.sb-data b{color:var(--green);}
.sb-note{margin-top:2rem; padding-top:1.1rem; border-top:1px solid var(--line-soft);
  color:var(--ink-3); font-size:.82rem; line-height:1.6;}

/* ── input ───────────────────────────────────────────── */
[data-testid="stChatInput"]{background:var(--surface); border:1px solid var(--line);
  border-radius:16px; box-shadow:0 10px 30px rgba(0,0,0,.45); transition:border-color .2s, box-shadow .2s;}
[data-testid="stChatInput"]:focus-within{border-color:var(--green);
  box-shadow:0 10px 34px rgba(0,0,0,.5), 0 0 0 1px rgba(63,191,127,.28), 0 0 30px -10px var(--green-glow);}
[data-testid="stChatInputTextArea"]{min-height:3rem!important; font-size:1.03rem; color:var(--ink);}
[data-testid="stBottomBlockContainer"]{background:transparent; padding-bottom:1.3rem;}
[data-testid="stExpander"]{border:none!important; box-shadow:none!important; background:transparent;}
[data-testid="stExpander"] summary{font-size:.9rem; color:var(--green); font-weight:500;}
[data-testid="stExpander"] summary:hover{color:#79D9A8;}
[data-testid="stSpinner"] > div{border-top-color:var(--green)!important;}

@media (prefers-reduced-motion: reduce){
  [data-testid="stAppViewContainer"],.hero .mark,.hero .mark span,.hero .sub,.hero .bar,
  .stButton button,[data-testid="stChatMessage"],.step{animation:none!important;}
  .stButton button:hover{transform:none;}}
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
                    f'<span class="src-doc">{html.escape(doc)}</span></div>'
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
if "history" not in st.session_state:
    st.session_state.history = []

# ── sidebar: identity, data source, disclaimer ────────────────────────────
with st.sidebar:
    # One wordmark on screen at a time: the hero carries it on the empty
    # state, the sidebar takes over once the conversation replaces the hero.
    if st.session_state.history:
        st.markdown('<div class="sb-mark">Scheme<span>Setu</span></div>',
                    unsafe_allow_html=True)
    st.markdown('<p class="sb-text">Grounded answers about Indian government '
                'schemes in English and Hindi.</p>'
                f'<div class="sb-data">Currently searching across <b>{n_docs} official '
                f'documents</b> — eligibility rules, benefits and required papers.</div>'
                '<p class="sb-note">Answers are generated by AI and are not professional '
                'advice. Always confirm details with the official scheme portal or your '
                'nearest government office before applying.</p>', unsafe_allow_html=True)

# ── main: header, suggestions, conversation ───────────────────────────────
picked = None
if not st.session_state.history:
    # The wordmark lives in the sidebar permanently, so the hero is the EMPTY
    # STATE only: a welcome before the first question, gone once the
    # conversation is the point of the screen. No title shown twice.
    st.markdown('<div class="hero"><div class="mark">Scheme<span>Setu</span></div>'
                '<p class="sub">Ask about any government scheme in English or हिंदी — '
                'eligibility, benefits, or the documents you need.</p>'
                '<div class="bar"></div></div>', unsafe_allow_html=True)
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
