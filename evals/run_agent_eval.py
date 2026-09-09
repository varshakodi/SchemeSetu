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
from slices import split  # noqa: E402

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


def outcome(state: dict) -> str:
    """What the agent actually did: 'answered', 'refused' or 'chitchat'.

    Read from the graph path, not from whether the reply contains "Sources:".
    That substring test was wrong in both directions -- a refusal explaining
    which sources it checked reads as an answer, and a chitchat reply reads as
    a refusal. The same bug shipped in the UI and was fixed there; the harness
    kept it, which is worse, because a harness reports a number rather than
    looking odd on screen.
    """
    last = state["path"][-1] if state.get("path") else ""
    if last == "refuse":
        return "refused"
    if last == "direct_reply":
        return "chitchat"
    return "answered"


def judge_ambiguous(judges: list, question: str, criterion: str, response: str):
    """Grade an underspecified question against the rubric carried in the row.

    Correctness here is a property of behaviour, not of content, so there is
    nothing to string-match: the question is whether the agent asked for what
    was missing or named the assumption it answered under.
    """
    for judge, name in list(judges):
        try:
            verdict = judge.complete(
                "You grade an assistant's handling of an underspecified "
                "question against a stated criterion. Reply PASS or FAIL on "
                "the first line, then one short sentence of reason.",
                f"Question asked:\n{question}\n\nCriterion:\n{criterion}\n\n"
                f"Assistant's reply:\n{response}",
                max_tokens=120)
            return verdict.strip().upper().startswith("PASS"), name, verdict.strip()
        except Exception:
            judges.remove((judge, name))
            continue
    return None, None, ""


def judge_faithfulness(judges: list, hits: list[dict], answer: str):
    """Return (True/False, judge_name, reason), or (None, None, "") if all down.

    The reason is kept because a faithfulness score without it is unactionable:
    when this metric missed its target, the first question was "which claim was
    unsupported?" and the harness had thrown that away.
    """
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
            return (verdict.strip().upper().startswith("PASS"), name,
                    " ".join(verdict.split())[:200])
        except Exception:
            # Dead for this run (daily quota, outage) — remove it so later
            # questions don't pay its failure latency again.
            judges.remove((judge, name))
            continue
    return None, None, ""


def evaluate(path: Path, app, judges: list) -> dict:
    rows = load_jsonl(path)
    answerable, traps, ambiguous = split(rows)
    kind = {r["id"]: k for group, k in
            ((answerable, "answerable"), (traps, "trap"), (ambiguous, "ambiguous"))
            for r in group}
    stats = {"traps": 0, "traps_refused": 0, "ans": 0, "false_refusals": 0,
             "cites_gold": 0, "faithful": 0, "answered": 0, "judged": 0,
             "rewrites": 0, "ambig": 0, "ambig_ok": 0, "ambig_judged": 0,
             "errors": 0, "attempted": 0}

    judges_used: set[str] = set()
    unfaithful: list[tuple[str, str, str]] = []
    print(f"\n=== {path.name} — full agent, judges={[n for _, n in judges]} ===")
    for r in rows:
        try:
            state = app.invoke({"question": r["question"], "path": []})
        except Exception as exc:
            # A free-tier daily quota can die mid-run. Losing every completed
            # question to one exception is unaffordable here: a full pass over
            # this set costs most of a day's token budget, so partial results
            # are worth keeping and labelling rather than discarding.
            stats["errors"] += 1
            print(f"  {r['id']:<9} STOPPED — {type(exc).__name__}: "
                  f"{' '.join(str(exc).split())[:160]}")
            stats["incomplete_after"] = r["id"]
            break
        stats["attempted"] += 1
        result = outcome(state)
        answered = result == "answered"
        rewrote = any(step.startswith("rewrite") for step in state["path"])
        stats["rewrites"] += rewrote

        if kind[r["id"]] == "trap":
            stats["traps"] += 1
            stats["traps_refused"] += (not answered)
            verdict = "refused ✓" if not answered else "ANSWERED ✗ (should refuse)"
        elif kind[r["id"]] == "ambiguous":
            stats["ambig"] += 1
            ok, judge_name, why = judge_ambiguous(
                judges, r["question"], r["answer"], state.get("response", ""))
            if ok is None:
                verdict = "ambiguous — unjudged (all judges down)"
            else:
                stats["ambig_judged"] += 1
                stats["ambig_ok"] += ok
                verdict = f"ambiguous — {'handled ✓' if ok else 'CONFIDENT ✗'} ({why.splitlines()[-1][:70]})"
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
                faithful, judge_name, reason = judge_faithfulness(
                    judges, state["hits"], state["response"])
                if judge_name:
                    judges_used.add(judge_name)
                if faithful is None:
                    f_mark = "unjudged (all judges down)"
                else:
                    stats["judged"] += 1
                    stats["faithful"] += faithful
                    f_mark = f"faithful={'✓' if faithful else '✗'}"
                    if not faithful:
                        unfaithful.append((r["id"], judge_name, reason))
                verdict = f"answered — cites_gold={'✓' if cited else '✗'} {f_mark}"
        rw = " [rewrote]" if rewrote else ""
        print(f"  {r['id']:<9} {verdict}{rw}")
        time.sleep(PAUSE_SECONDS)

    if stats.get("errors"):
        print(f"\n  INCOMPLETE RUN: stopped at {stats.get('incomplete_after')} "
              f"after {stats['attempted']}/{len(rows)} questions. The rates below "
              f"are computed over what completed and are NOT comparable with a "
              f"full run — re-run when quota allows.")
    if stats["traps"]:
        print(f"\n  trap refusal   = {stats['traps_refused']}/{stats['traps']}")
    if stats["ambig"]:
        print(f"  ambiguous ok   = {stats['ambig_ok']}/{stats['ambig_judged']} "
              f"({stats['ambig'] - stats['ambig_judged']} unjudged)")
    if stats["ans"]:
        print(f"  false refusal  = {stats['false_refusals']}/{stats['ans']}")
    if stats["answered"]:
        print(f"  cites gold doc = {stats['cites_gold']}/{stats['answered']}")
        print(f"  faithfulness   = {stats['faithful']}/{stats['judged']} "
              f"({stats['answered'] - stats['judged']} unjudged)")
        if unfaithful:
            print("\n  judged unfaithful:")
            for qid, jname, reason in unfaithful:
                print(f"    {qid}  [{jname}]  {reason[:150]}")
        if len(judges_used) > 1:
            print(
                "\n  WARNING: more than one judge ran during this file "
                f"({', '.join(sorted(judges_used))}). A free-tier judge that dies "
                "mid-run is silently replaced by the next one, so the faithfulness "
                "score above mixes judges of different strictness and is NOT a "
                "single measurement. Re-run pinned to one judge: --judge <name>."
            )
    print(f"  rewrite loop used on {stats['rewrites']} question(s)")
    return stats


def main() -> None:
    args = sys.argv[1:]
    pinned = None
    if "--judge" in args:
        i = args.index("--judge")
        pinned = args[i + 1]
        del args[i:i + 2]
    files = [Path(a) for a in args] or [ROOT / "evals" / "golden_set.jsonl",
                                        ROOT / "evals" / "dev_set.jsonl"]
    judges = make_judges()
    if pinned:
        judges = [(j, n) for j, n in judges if pinned in n]
        if not judges:
            sys.exit(f"No judge matching {pinned!r}. Available: "
                     f"{[n for _, n in make_judges()]}")
        # Pinned means pinned: no silent substitution if this one dies. A run
        # with a consistent judge and gaps beats one with a hidden handover.
        print(f"judge pinned to {judges[0][1]} — no fallback")
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
    if totals.get("ambig_judged"):
        rate = totals["ambig_ok"] / totals["ambig_judged"]
        print(f"  ambiguous     {rate:.2f}  (no PRD target — measures a known "
              f"gap: the agent has no clarify path)")
    if totals.get("judged"):
        print(f"  faithfulness  {totals['faithful'] / totals['judged']:.2f}  "
              f"on {totals['judged']}/{totals['answered']} judged "
              f"(target >= 0.90 once measured on the frozen golden set)")


if __name__ == "__main__":
    main()
