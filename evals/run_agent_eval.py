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
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.graph import build_graph  # noqa: E402
from agent.llm import (  # noqa: E402
    PROVIDERS, USAGE, OpenAICompatLLM, available_provider, get_llm, reset_usage)
from naive.rag import build_context  # noqa: E402
from slices import split  # noqa: E402

# Free tiers meter tokens per MINUTE, not just per day: Groq allows 8,000 TPM
# on the generator, and one question costs roughly 2,000 across classify,
# generate and verify. That is about four questions a minute, so a 2 s pause
# guaranteed a 429 every few questions -- survivable now that the backoff
# classifies throttles correctly, but each one is a wasted round-trip. Pace
# instead of retrying. Override with SCHEMESETU_EVAL_PAUSE when a paid tier
# or a different provider makes this unnecessary.
PAUSE_SECONDS = float(os.environ.get("SCHEMESETU_EVAL_PAUSE", 14.0))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def make_judges() -> list[tuple]:
    """Judges in preference order, most independent first.

    "Independent" means a different MODEL, not merely a different vendor's
    API. Judging gpt-oss-120b's output with gpt-oss-120b served by someone
    else is self-judging with extra steps -- same weights, same blind spots.
    So the ordering is by model family first, provider second.

    Free-tier facts, checked 9 Sept 2026 (these age -- re-check on failure):
      gemini-3.6-flash   different provider AND family. Best evidence, but
                         ~20 requests/day, short of the ~50 a full golden
                         run needs -- it will die mid-run and be replaced.
      qwen3.8-27b        different family, same provider as the generator.
                         Groq meters tokens per DAY PER MODEL, so this does
                         not compete with the generator's budget -- which is
                         what makes a full pinned run finish at all.
      gpt-oss-20b        same family, smaller. Weakest; last resort.
      cerebras           deliberately absent: every model there, gpt-oss-120b
                         included, returns 402 payment_required. No free tier.
    """
    import os
    generator = available_provider()
    judges = []

    if generator != "gemini" and os.environ.get("GEMINI_API_KEY"):
        cfg = PROVIDERS["gemini"]
        judges.append((OpenAICompatLLM(cfg["base_url"], os.environ["GEMINI_API_KEY"],
                                       cfg["model"]),
                       "gemini-3.6-flash (different provider and family)"))
    if os.environ.get("GROQ_API_KEY"):
        cfg = PROVIDERS["groq"]
        judges.append((OpenAICompatLLM(cfg["base_url"], os.environ["GROQ_API_KEY"],
                                       "qwen/qwen3.8-27b"),
                       "qwen3.8-27b (different family, own token budget)"))
        judges.append((OpenAICompatLLM(cfg["base_url"], os.environ["GROQ_API_KEY"],
                                       "openai/gpt-oss-20b"),
                       "gpt-oss-20b (same family as generator -- weakest)"))
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


def judge_relevance(judges: list, question: str, answer: str):
    """Does the answer address the question that was asked? (PRD §5, >= 0.85)

    Distinct from faithfulness. An answer can be perfectly grounded in the
    retrieved passages and still answer a different question than the user
    asked -- that is what the near-miss traps look like when they slip
    through. Faithfulness checks claims against context; this checks the
    answer against the question.
    """
    for judge, name in list(judges):
        try:
            verdict = judge.complete(
                "Reply PASS if the answer addresses the question that was "
                "asked, or appropriately declines when it cannot. Reply FAIL "
                "if it answers a different question. First line PASS or FAIL.",
                f"Question:\n{question}\n\nAnswer:\n{answer}", max_tokens=80)
            return verdict.strip().upper().startswith("PASS")
        except Exception:
            judges.remove((judge, name))
            continue
    return None


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


def results_path(path: Path) -> Path:
    return ROOT / "evals" / "runs" / f"{path.stem}.jsonl"


def load_done(path: Path) -> dict[str, dict]:
    """Per-question results already recorded, keyed by question id.

    A full pass costs about 140k generator tokens against a free tier that
    allows 200k a day and refills on a rolling window, so a run interrupted by
    quota cannot simply be restarted -- it would spend the next day's budget
    redoing work. Completed questions are written as they finish and skipped
    on resume. The system and the pinned judge are unchanged between sessions;
    the assembled date range is reported so the reader can see it was not one
    sitting.
    """
    p = results_path(path)
    if not p.exists():
        return {}
    return {r["id"]: r for r in
            (json.loads(line) for line in p.read_text().splitlines() if line.strip())}


def record(path: Path, row: dict) -> None:
    p = results_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate(path: Path, app, judges: list, resume: bool = False) -> dict:
    rows = load_jsonl(path)
    done = load_done(path) if resume else {}
    answerable, traps, ambiguous = split(rows)
    kind = {r["id"]: k for group, k in
            ((answerable, "answerable"), (traps, "trap"), (ambiguous, "ambiguous"))
            for r in group}

    judges_used: set[str] = set()
    recs: list[dict] = []
    stopped_at = None

    print(f"\n=== {path.name} — full agent, judges={[n for _, n in judges]} ===")
    if done:
        print(f"  resuming: {len(done)} already recorded, re-using them")

    for r in rows:
        if r["id"] in done:
            recs.append(done[r["id"]])
            continue

        reset_usage()
        try:
            state = app.invoke({"question": r["question"], "path": []})
        except Exception as exc:
            # A free-tier daily budget dies mid-run, and a full pass costs most
            # of one. Stop cleanly; everything already recorded survives on
            # disk and --resume picks up here rather than paying for it twice.
            print(f"  {r['id']:<9} STOPPED — {type(exc).__name__}: "
                  f"{' '.join(str(exc).split())[:400]}")
            stopped_at = r["id"]
            break
        gen_tokens = USAGE["prompt"] + USAGE["completion"]

        rec = {"id": r["id"], "kind": kind[r["id"]], "outcome": outcome(state),
               "rewrote": any(s.startswith("rewrite") for s in state["path"]),
               "gen_tokens": gen_tokens}
        answered = rec["outcome"] == "answered"

        if rec["kind"] == "trap":
            rec["ok"] = not answered
            verdict = "refused ✓" if rec["ok"] else "ANSWERED ✗ (should refuse)"
        elif rec["kind"] == "ambiguous":
            ok, jname, why = judge_ambiguous(
                judges, r["question"], r["answer"], state.get("response", ""))
            rec["ok"], rec["reason"] = ok, why[:200]
            if jname:
                judges_used.add(jname)
            verdict = ("ambiguous — unjudged" if ok is None else
                       f"ambiguous — {'handled ✓' if ok else 'CONFIDENT ✗'}")
        else:
            rec["answered"] = answered
            if answered:
                rec["cites_gold"] = any(
                    g in state["response"].split("Sources:")[-1]
                    for g in r["gold_doc_ids"])
                faithful, jname, reason = judge_faithfulness(
                    judges, state["hits"], state["response"])
                rec["faithful"], rec["reason"] = faithful, reason[:200]
                rec["relevant"] = judge_relevance(
                    judges, r["question"], state["response"])
                if jname:
                    judges_used.add(jname)
                    rec["judge"] = jname
                verdict = (f"answered — cites_gold={'✓' if rec['cites_gold'] else '✗'} "
                           f"faithful={'?' if faithful is None else '✓' if faithful else '✗'} "
                           f"relevant={'?' if rec['relevant'] is None else '✓' if rec['relevant'] else '✗'}")
            else:
                verdict = "REFUSED ✗ (should answer)"

        rec["judge_tokens"] = USAGE["prompt"] + USAGE["completion"] - gen_tokens
        recs.append(rec)
        record(path, rec)
        print(f"  {r['id']:<9} {verdict}{' [rewrote]' if rec['rewrote'] else ''}")
        time.sleep(PAUSE_SECONDS)

    return summarise(path, rows, recs, judges_used, stopped_at)


def summarise(path, rows, recs, judges_used, stopped_at) -> dict:
    """Aggregate whatever has been recorded, resumed entries included."""
    def where(**kw):
        return [r for r in recs if all(r.get(k) == v for k, v in kw.items())]

    traps = where(kind="trap")
    ambig = where(kind="ambiguous")
    ans = [r for r in recs if r["kind"] == "answerable"]
    answered = [r for r in ans if r.get("answered")]
    judged_f = [r for r in answered if r.get("faithful") is not None]
    judged_r = [r for r in answered if r.get("relevant") is not None]
    judged_a = [r for r in ambig if r.get("ok") is not None]

    if stopped_at or len(recs) < len(rows):
        print(f"\n  INCOMPLETE: {len(recs)}/{len(rows)} recorded"
              + (f", stopped at {stopped_at}" if stopped_at else "")
              + ". Re-run with --resume to continue; recorded questions are "
                "kept and will not be paid for twice.")

    if traps:
        print(f"\n  trap refusal   = {sum(r['ok'] for r in traps)}/{len(traps)}")
    if judged_a:
        print(f"  ambiguous ok   = {sum(bool(r['ok']) for r in judged_a)}/{len(judged_a)}")
    if ans:
        print(f"  false refusal  = {len(ans) - len(answered)}/{len(ans)}")
    if answered:
        print(f"  cites gold doc = {sum(bool(r.get('cites_gold')) for r in answered)}/{len(answered)}")
    if judged_f:
        print(f"  faithfulness   = {sum(r['faithful'] for r in judged_f)}/{len(judged_f)}")
        bad = [r for r in judged_f if not r["faithful"]]
        if bad:
            print("\n  judged unfaithful:")
            for r in bad:
                print(f"    {r['id']}  [{r.get('judge','?')}]  {r.get('reason','')[:140]}")
    if judged_r:
        print(f"  answer relevance = {sum(r['relevant'] for r in judged_r)}/{len(judged_r)}")

    tokens = sum(r.get("gen_tokens", 0) + r.get("judge_tokens", 0) for r in recs)
    if tokens and recs:
        print(f"\n  tokens = {tokens:,} over {len(recs)} questions "
              f"({tokens / len(recs):,.0f}/question) · cost ₹0 (free tiers)")
    print(f"  rewrite loop used on {sum(bool(r.get('rewrote')) for r in recs)} question(s)")
    if len(judges_used) > 1:
        print(f"\n  WARNING: more than one judge ran ({', '.join(sorted(judges_used))}). "
              "The scores above mix judges of different strictness and are NOT a "
              "single measurement. Re-run pinned with --judge <name>.")

    return {"recorded": len(recs), "total": len(rows),
            "traps": len(traps), "traps_refused": sum(r["ok"] for r in traps),
            "ans": len(ans), "false_refusals": len(ans) - len(answered),
            "answered": len(answered),
            "cites_gold": sum(bool(r.get("cites_gold")) for r in answered),
            "judged": len(judged_f), "faithful": sum(r["faithful"] for r in judged_f),
            "rel_judged": len(judged_r), "relevant": sum(r["relevant"] for r in judged_r),
            "ambig_judged": len(judged_a), "ambig_ok": sum(bool(r["ok"]) for r in judged_a),
            "tokens": tokens}


def main() -> None:
    args = sys.argv[1:]
    resume = "--resume" in args
    if resume:
        args.remove("--resume")
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
        for k, v in evaluate(f, app, judges, resume=resume).items():
            if isinstance(v, (int, float)):   # 'incomplete_after' holds a question id
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
        rate = totals["faithful"] / totals["judged"]
        print(f"  faithfulness  {rate:.2f}  (target >= 0.90) "
              f"{'✓' if rate >= 0.90 else '✗'}  on {totals['judged']}/"
              f"{totals['answered']} answered")
    if totals.get("rel_judged"):
        rate = totals["relevant"] / totals["rel_judged"]
        print(f"  answer relev. {rate:.2f}  (target >= 0.85) "
              f"{'✓' if rate >= 0.85 else '✗'}  on {totals['rel_judged']} answers")
    if totals.get("tokens") and totals.get("recorded"):
        per = totals["tokens"] / totals["recorded"]
        print(f"  cost          ₹0 (free tiers) · {per:,.0f} tokens/question, "
              f"{totals['tokens']:,} total")
    if totals.get("recorded", 0) < totals.get("total", 0):
        print(f"\n  {totals['recorded']}/{totals['total']} questions recorded — "
              "re-run with --resume when the token budget refills.")


if __name__ == "__main__":
    main()
