"""Agent graph tests — routing logic with a scripted LLM.

The graph's decisions (route on label, on evidence, on verification) are pure
logic; only the node *contents* need an LLM. So we inject a ScriptedLLM and a
fake retriever, and test the state machine's behaviour offline — including the
loop that rescues dev-012. An integration variant with REAL retrieval runs
when sentence-transformers is installed (locally; skipped in CI).
"""

import pytest

from agent.graph import build_graph
from naive.rag import REFUSAL_THRESHOLD

STRONG, WEAK = REFUSAL_THRESHOLD + 5.0, REFUSAL_THRESHOLD - 5.0


class ScriptedLLM:
    """Returns canned responses by matching a keyword in the system prompt."""

    def __init__(self, script: dict[str, str]):
        self.script, self.calls = script, []

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        for marker, reply in self.script.items():
            if marker in system:
                self.calls.append(marker)
                return reply
        raise AssertionError(f"no scripted reply matches system prompt: {system[:60]}")


def fake_retriever(scores_by_query: dict[str, float]):
    def retrieve(query: str, k: int = 4):
        score = scores_by_query.get(query, WEAK)
        return [{"doc_id": "doc.md", "chunk_id": "doc.md#0",
                 "text": "chunk text", "rerank": score, "score": score,
                 "cosine": 0.5}]
    return retrieve


def run(llm, retriever, question="q?"):
    app = build_graph(llm=llm, retriever=retriever)
    return app.invoke({"question": question, "path": []})


def test_chitchat_skips_retrieval():
    llm = ScriptedLLM({"Classify": "chitchat", "Reply warmly": "Hello! Ask me about schemes."})
    state = run(llm, fake_retriever({}), "hi there!")
    assert "Hello" in state["response"]
    assert not any(step.startswith("retrieve") for step in state["path"])


def test_out_of_domain_refuses_without_retrieval():
    llm = ScriptedLLM({"Classify": "out_of_domain"})
    state = run(llm, fake_retriever({}), "write my DSA homework")
    assert "don't know" in state["response"]
    assert not any(step.startswith("retrieve") for step in state["path"])


def test_strong_evidence_generates_verified_cited_answer():
    llm = ScriptedLLM({"Classify": "in_domain",
                       "SchemeSetu, an assistant for Indian government schemes.\nAnswer ONLY": "Rs 6000 per year [1].",
                       "fact-checker": "PASS"})
    state = run(llm, fake_retriever({"q?": STRONG}))
    assert "Rs 6000" in state["response"] and "Sources:" in state["response"]


def test_weak_evidence_rewrites_then_answers():
    """The dev-012 rescue: weak on the raw query, strong on the rewrite."""
    llm = ScriptedLLM({"Classify": "in_domain",
                       "Rewrite": "maternity benefit pregnant women",
                       "SchemeSetu, an assistant for Indian government schemes.\nAnswer ONLY": "Rs 5000 in two instalments [1].",
                       "fact-checker": "PASS"})
    retriever = fake_retriever({"q?": WEAK, "maternity benefit pregnant women": STRONG})
    state = run(llm, retriever)
    assert "Rs 5000" in state["response"]
    assert any(step.startswith("rewrite") for step in state["path"])


def test_weak_twice_refuses_honestly():
    llm = ScriptedLLM({"Classify": "in_domain", "Rewrite": "still nothing useful"})
    state = run(llm, fake_retriever({}))  # every query scores WEAK
    assert "don't know" in state["response"]
    assert sum(step.startswith("retrieve") for step in state["path"]) == 2  # tried twice


def test_failed_grounding_regenerates_then_refuses():
    llm = ScriptedLLM({"Classify": "in_domain",
                       "SchemeSetu, an assistant for Indian government schemes.\nAnswer ONLY": "Made-up claim [1].",
                       "fact-checker": "FAIL unsupported amount"})
    state = run(llm, fake_retriever({"q?": STRONG}))
    assert "don't know" in state["response"]
    assert sum(step == "generate" for step in state["path"]) == 2  # original + one regen


try:
    import sentence_transformers  # noqa: F401
    HAS_ML = True
except ImportError:
    HAS_ML = False


@pytest.mark.skipif(not HAS_ML, reason="needs sentence-transformers (skipped in CI)")
def test_integration_real_retrieval_dev012_rescue():
    """Real index + real reranker; only the LLM is scripted. Proves the rewrite
    turns dev-012's below-threshold score into an above-threshold one."""
    from naive.rag import search
    llm = ScriptedLLM({"Classify": "in_domain",
                       "Rewrite": "PMMVY maternity benefit amount for pregnant women first child",
                       "SchemeSetu, an assistant for Indian government schemes.\nAnswer ONLY": "Rs 5,000 in two instalments [1].",
                       "fact-checker": "PASS"})
    app = build_graph(llm=llm, retriever=search)
    state = app.invoke({"question": "My wife is expecting our first baby. "
                                    "Is there any government money we can get?",
                        "path": []})
    assert "Sources:" in state["response"]          # answered, not refused
    assert any(step.startswith("rewrite") for step in state["path"])
    assert "myscheme_pmmvy.md" in state["response"]  # cited the right document
