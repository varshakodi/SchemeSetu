"""SchemeSetu — the agent's final exam: every question through the FULL graph.

`run_eval.py` measures retrieval in isolation. This harness measures what a
user actually experiences — the complete agent (classify → retrieve → rewrite?
→ generate → verify → answer/refuse), live LLM calls included — and scores:

  trap refusal    Did the agent refuse every unanswerable question?  (>= 0.90)
  false refusal   Did it wrongly refuse answerable ones?             (<= 0.10)
  cites gold      When it answered, do the cited sources include a
                  document we know contains the answer?
  faithfulness    LLM-AS-JUDGE: an *independent* model (a different provider
                  than the generator — PRD §8's judge rule, mapped onto two
                  free tiers) checks every claim in the answer against the
                  retrieved context. Judging your own homework is worthless;
                  independence is what makes the number mean something.

Free-tier etiquette: one question at a time with a courtesy pause — plus the
429 backoff in agent/llm.py when the meter trips anyway.

Usage:  python evals/run_agent_eval.py            # both sets
        python evals/run_agent_eval.py evals/dev_set.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.graph import build_graph  # noqa: E402
from agent.llm import PROVIDERS, OpenAICompatLLM, available_provider, get_llm  # noqa: E402
from naive.rag import build_context  # noqa: E402

PAUSE_SECONDS = 2.0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def make_judges() -> list[tuple]:
    """Judges in preference order. Best: a DIFFERENT provider than the
    generator (true independence). Fallback: same provider, different MODEL
    (gpt-oss-20b judging gpt-oss-120b) — weaker independence, honestly
    labelled. Free-tier quotas are per-day (Gemini: ~20 requests), so a judge
    can die mid-run; the harness downgrades instead of crashing."""
    import os
    generator = available_provider()
    judges = []
    if generator != "gemini" and os.environ.get("GEMINI_API_KEY"):
        cfg = PROVIDERS["gemini"]
        judges.append((OpenAICompatLLM(cfg["base_url"], os.environ["GEMINI_API_KEY"],
                                       cfg["model"]), "gemini"))
    if os.environ.get("GROQ_API_KEY"):
        cfg = PROVIDERS["groq"]
        judges.append((OpenAICompatLLM(cfg["base_url"], os.environ["GROQ_API_KEY"],
                                       "openai/gpt-oss-20b"),
                       "groq:gpt-oss-20b (same-provider fallback)"))
    if not judges:
        judges.append((get_llm(), "generator itself (weakest evidence)"))
    return judges


def judge_faithfulness(judges: list, hits: list[dict], answer: str):
    """Return (True/False, judge_name) or (None, None) if every judge is down."""
    for judge, name in list(judges):
        try:
            verdict = judge.complete(
                "You are a strict, independent fact-checker. Given context "
                "passages and an answer, reply PASS if every factual claim in "
                "the answer is supported by the context; otherwise reply FAIL "
                "plus the unsupported claim. Citations like [1] refer to the "
                "numbered passages.",
                f"Context passages:\n{build_context(hits)}\n\nAnswer:\n{answer}",
                max_tokens=150)
            return verdict.strip().upper().startswith("PASS"), name
        except Exception:
            # Dead for this run (daily quota, outage) — remove it so later
            # questions don't pay its failure latency again.
            judges.remove((judge, name))
            continue
    return None, None


def evaluate(path: Path, app, judges: list) -> dict:
    rows = load_jsonl(path)
    stats = {"traps": 0, "traps_refused": 0, "ans": 0, "false_refusals": 0,
             "cites_gold": 0, "faithful": 0, "answered": 0, "judged": 0,
             "rewrites": 0}

    print(f"\n=== {path.name} — full agent, judges={[n for _, n in judges]} ===")
    for r in rows:
        state = app.invoke({"question": r["question"], "path": []})
        answered = "Sources:" in state.get("response", "")
        rewrote = any(step.startswith("rewrite") for step in state["path"])
        stats["rewrites"] += rewrote

        if not r.get("gold_doc_ids"):  # trap
            stats["traps"] += 1
            stats["traps_refused"] += (not answered)
            verdict = "refused ✓" if not answered else "ANSWERED ✗ (should refuse)"
        else:
            stats["ans"] += 1
            if not answered:
                stats["false_refusals"] += 1
                verdict = "REFUSED ✗ (should answer)"
            else:
                stats["answered"] += 1
                cited = any(g in state["response"].split("Sources:")[-1]
                            for g in r["gold_doc_ids"])
                stats["cites_gold"] += cited
                faithful, judge_name = judge_faithfulness(
                    judges, state["hits"], state["response"])
                if faithful is None:
                    f_mark = "unjudged (all judges down)"
                else:
                    stats["judged"] += 1
                    stats["faithful"] += faithful
                    f_mark = f"faithful={'✓' if faithful else '✗'}"
                verdict = f"answered — cites_gold={'✓' if cited else '✗'} {f_mark}"
        rw = " [rewrote]" if rewrote else ""
        print(f"  {r['id']:<9} {verdict}{rw}")
        time.sleep(PAUSE_SECONDS)

    if stats["traps"]:
        print(f"\n  trap refusal   = {stats['traps_refused']}/{stats['traps']}")
    if stats["ans"]:
        print(f"  false refusal  = {stats['false_refusals']}/{stats['ans']}")
    if stats["answered"]:
        print(f"  cites gold doc = {stats['cites_gold']}/{stats['answered']}")
        print(f"  faithfulness   = {stats['faithful']}/{stats['judged']} "
              f"({stats['answered'] - stats['judged']} unjudged)")
    print(f"  rewrite loop used on {stats['rewrites']} question(s)")
    return stats


def main() -> None:
    files = [Path(a) for a in sys.argv[1:]] or [ROOT / "evals" / "golden_set.jsonl",
                                               ROOT / "evals" / "dev_set.jsonl"]
    judges = make_judges()
    app = build_graph()
    totals: dict[str, int] = {}
    for f in files:
        for k, v in evaluate(f, app, judges).items():
            totals[k] = totals.get(k, 0) + v

    print("\n=== overall vs PRD §5 targets ===")
    if totals.get("traps"):
        rate = totals["traps_refused"] / totals["traps"]
        print(f"  trap refusal  {rate:.2f}  (target >= 0.90) "
              f"{'✓' if rate >= 0.90 else '✗'}")
    if totals.get("ans"):
        rate = totals["false_refusals"] / totals["ans"]
        print(f"  false refusal {rate:.2f}  (target <= 0.10) "
              f"{'✓' if rate <= 0.10 else '✗'}")
    if totals.get("judged"):
        print(f"  faithfulness  {totals['faithful'] / totals['judged']:.2f}  "
              f"on {totals['judged']}/{totals['answered']} judged "
              f"(target >= 0.90 once measured on the frozen golden set)")


if __name__ == "__main__":
    main()
