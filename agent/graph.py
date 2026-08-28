"""SchemeSetu — Week 5: the agent. A LangGraph state machine around retrieval.

WHY AN AGENT AT ALL? Weeks 1-4 built a *pipeline*: a straight line from
question to answer. A pipeline cannot make decisions — it can't say "this
question is small talk, skip retrieval", or "the evidence looks weak, let me
rephrase and try again", or "my draft answer isn't supported, regenerate".
An agent is a STATE MACHINE: nodes do work, conditional edges choose the next
node based on state, and CYCLES let the system retry. That last part is the
whole point — LangGraph's contribution over a plain chain is loops, branching,
and checkpointed state (conversation memory).

The graph (PRD §7):

    classify ──chitchat──────────────► direct_reply ─► END
        │ out_of_domain ─────────────► refuse ───────► END
        │ in_domain
        ▼
    retrieve ──strong evidence──────► generate ─► verify ──pass──► END
        │ weak, first try                 ▲           │ fail, first try
        ▼                                 └───────────┘ (regenerate once)
    rewrite ──► (back to retrieve)                    │ fail again
        weak again ─────────────────► refuse ◄────────┘

Design choices worth defending in an interview (decisions.md 012):

- EVIDENCE GRADING IS THE RERANK SCORE. The PRD calls for "CRAG-lite" grading
  of retrieved evidence. We already have a tuned, threshold-calibrated
  relevance signal — the cross-encoder score from Week 4 — so grading costs
  zero extra LLM calls. (CRAG papers use an LLM grader; ours is a measured
  substitute, revisit if the golden set says otherwise.)

- THE REWRITE LOOP EXISTS BECAUSE OF dev-012. "My wife is expecting our first
  baby..." retrieves the RIGHT document but scores -6.07 (conversational
  paraphrase, zero lexical overlap). One rewrite into scheme-register language
  ("maternity benefit for pregnant women") turns a wrong refusal into a
  correct, cited answer. The threshold stopped being a wall and became a door.

- LLM CALLS GO THROUGH A TINY SEAM (`LLM.complete`). Nodes never import the
  Anthropic SDK directly; they call an injected `llm`. Production uses
  `ClaudeLLM`; tests inject a scripted fake — so the graph's routing logic is
  fully testable offline, with REAL retrieval underneath.

Usage (needs ANTHROPIC_API_KEY for the LLM nodes):
    python agent/graph.py ask "Am I eligible for PM-KISAN?"
    python agent/graph.py ask --trace "What subsidy for an electric scooter?"

(Conversation memory — checkpointed threads with message history — is FR-5.2,
landing separately: build_graph already accepts a checkpointer for it.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from naive.rag import (  # noqa: E402
    REFUSAL_THRESHOLD, SYSTEM_PROMPT, build_context, search,
)
from agent.llm import get_llm  # noqa: E402  (the provider seam — see agent/llm.py)

MAX_REWRITES = 1     # one rewrite-and-retry before refusing (PRD FR-5.1)
MAX_REGENS = 1       # one regeneration after a failed grounding check


# --- State --------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    question: str      # what the user asked
    label: str         # chitchat | in_domain | out_of_domain
    query: str         # the string we actually retrieve with (rewritten on retry)
    rewrites: int
    hits: list         # retrieved chunks (with rerank scores)
    evidence: float    # top-1 rerank score = our evidence grade
    answer: str        # draft answer
    regens: int
    response: str      # final user-facing text
    path: list         # node names visited, for the --trace view


# --- Graph construction -------------------------------------------------------

def build_graph(llm=None, checkpointer=None, retriever=None):
    """Wire the state machine. `llm` and `retriever` are injectable for tests."""
    llm = llm if llm is not None else get_llm()
    retrieve_fn = retriever if retriever is not None else search

    def classify(state: AgentState) -> AgentState:
        """Route the question: not everything deserves retrieval."""
        label = llm.complete(
            "Classify the user's message for an assistant that answers questions "
            "about Indian government welfare schemes using an official-document "
            "corpus. Reply with exactly one word:\n"
            "chitchat — greetings, thanks, small talk about the assistant\n"
            "in_domain — anything plausibly about government schemes, benefits, "
            "eligibility, subsidies, welfare, or documents for them\n"
            "out_of_domain — everything else (coding help, movies, homework...)",
            state["question"], max_tokens=8).strip().lower()
        if label not in {"chitchat", "in_domain", "out_of_domain"}:
            label = "in_domain"  # when unsure, try to help — refusal comes later anyway
        return {"label": label, "query": state["question"],
                "path": state.get("path", []) + ["classify"]}

    def direct_reply(state: AgentState) -> AgentState:
        text = llm.complete(
            "You are SchemeSetu, an assistant for Indian government schemes. "
            "Reply warmly in one or two sentences, and mention what you can help "
            "with. Do not invent scheme facts.", state["question"], max_tokens=150)
        return {"response": text, "path": state["path"] + ["direct_reply"]}

    def retrieve(state: AgentState) -> AgentState:
        hits = retrieve_fn(state["query"])
        evidence = hits[0].get("rerank", 0.0) if hits else float("-inf")
        return {"hits": hits, "evidence": evidence,
                "path": state["path"] + [f"retrieve(evidence={evidence:+.2f})"]}

    def rewrite(state: AgentState) -> AgentState:
        """Turn conversational phrasing into scheme-register search language."""
        new_q = llm.complete(
            "Rewrite the user's question as a short search query using the formal "
            "vocabulary of Indian government scheme documents (scheme names, "
            "benefit types, beneficiary categories). Reply with the query only.",
            state["question"], max_tokens=60)
        return {"query": new_q, "rewrites": state.get("rewrites", 0) + 1,
                "path": state["path"] + [f"rewrite({new_q[:40]!r})"]}

    def generate(state: AgentState) -> AgentState:
        # Second refusal layer: the retrieval threshold judges TOPICAL
        # closeness, but near-miss chunks (PMAY-G for a PMAY-Urban question)
        # can score high while lacking the actual answer. The generator reads
        # the content, so it gets a machine-readable way to say so.
        answer = llm.complete(
            SYSTEM_PROMPT + "\nIf the passages do not contain the answer to "
            "this question, reply with exactly NO_ANSWER and nothing else.",
            f"Context passages:\n\n{build_context(state['hits'])}\n\n"
            f"Question: {state['question']}")
        return {"answer": answer, "regens": state.get("regens", 0),
                "path": state["path"] + ["generate"]}

    def verify(state: AgentState) -> AgentState:
        """Grounding self-check: is every claim supported by the context?"""
        verdict = llm.complete(
            "You are a strict fact-checker. Given context passages and an answer, "
            "reply PASS if every factual claim in the answer is supported by the "
            "context AND the answer addresses the question it claims to answer. "
            "Reply FAIL (plus the reason) if any claim is unsupported, or if the "
            "answer is about a different scheme than the one asked about.",
            f"Context:\n{build_context(state['hits'])}\n\nAnswer:\n{state['answer']}",
            max_tokens=100)
        ok = verdict.strip().upper().startswith("PASS")
        step = "verify(pass)" if ok else f"verify(fail: {verdict[:40]!r})"
        out: AgentState = {"path": state["path"] + [step]}
        if ok:
            sources = "\n".join(f"  [{i + 1}] {h['chunk_id']}"
                                for i, h in enumerate(state["hits"]))
            out["response"] = f"{state['answer']}\n\nSources:\n{sources}"
        else:
            out["regens"] = state.get("regens", 0) + 1
        return out

    def refuse(state: AgentState) -> AgentState:
        if state.get("label") == "out_of_domain":
            why = "That is outside what I cover — Indian government welfare schemes."
        elif state.get("answer", "").strip().upper().startswith("NO_ANSWER"):
            why = ("The documents I found are topically close, but none of them "
                   "actually contains this answer.")
        else:
            why = (f"The best evidence I found scores {state.get('evidence', 0):+.2f}, "
                   f"below my confidence threshold ({REFUSAL_THRESHOLD}), even after "
                   f"rephrasing the search. My corpus most likely does not answer this.")
        return {"response": f"I don't know. {why}\nTry asking about a specific "
                            f"scheme's eligibility, benefits, or documents.",
                "path": state["path"] + ["refuse"]}

    # --- routers: the conditional edges ---
    def route_label(state: AgentState) -> str:
        return state["label"]

    def route_evidence(state: AgentState) -> str:
        if state["evidence"] >= REFUSAL_THRESHOLD:
            return "strong"
        return "retry" if state.get("rewrites", 0) < MAX_REWRITES else "give_up"

    def route_generated(state: AgentState) -> str:
        if state["answer"].strip().upper().startswith("NO_ANSWER"):
            return "no_answer"
        return "verify"

    def route_verified(state: AgentState) -> str:
        if "response" in state and state["response"]:
            return "done"
        return "regen" if state["regens"] <= MAX_REGENS else "give_up"

    g = StateGraph(AgentState)
    for name, fn in [("classify", classify), ("direct_reply", direct_reply),
                     ("retrieve", retrieve), ("rewrite", rewrite),
                     ("generate", generate), ("verify", verify),
                     ("refuse", refuse)]:
        g.add_node(name, fn)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_label, {
        "chitchat": "direct_reply", "in_domain": "retrieve",
        "out_of_domain": "refuse"})
    g.add_conditional_edges("retrieve", route_evidence, {
        "strong": "generate", "retry": "rewrite", "give_up": "refuse"})
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("generate", route_generated, {
        "no_answer": "refuse", "verify": "verify"})
    g.add_conditional_edges("verify", route_verified, {
        "done": END, "regen": "generate", "give_up": "refuse"})
    g.add_edge("direct_reply", END)
    g.add_edge("refuse", END)

    return g.compile(checkpointer=checkpointer)


# --- CLI ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="run one question through the agent")
    ask.add_argument("question")
    ask.add_argument("--trace", action="store_true",
                     help="print the path taken through the graph")
    args = parser.parse_args()

    app = build_graph()  # exits with provider setup guidance if none configured
    state = app.invoke({"question": args.question, "path": []})

    if args.trace:
        print("Trace:", "  ->  ".join(state["path"]), "\n" + "-" * 60)
    print(state["response"])


if __name__ == "__main__":
    main()
