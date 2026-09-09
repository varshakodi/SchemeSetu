"""Resume must not lose or double-count results.

A full pass over the golden set costs about 140k generator tokens against a
free tier of 200k per day that refills on a rolling window. A run interrupted
by quota therefore cannot simply be restarted -- that would spend the next
day's budget redoing work already paid for. Completed questions are appended
to evals/runs/<set>.jsonl as they finish and skipped on --resume.

These tests cover the file round-trip and the aggregation, which are pure and
need no LLM. The graded content itself is exercised by test_agent.py.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

from run_agent_eval import load_done, record, results_path, summarise  # noqa: E402


def test_record_then_load_round_trips(tmp_path, monkeypatch):
    import run_agent_eval as h
    monkeypatch.setattr(h, "ROOT", tmp_path)
    src = tmp_path / "some_set.jsonl"
    record(src, {"id": "g-001", "kind": "answerable", "answered": True})
    record(src, {"id": "g-002", "kind": "trap", "ok": True})
    done = load_done(src)
    assert set(done) == {"g-001", "g-002"}
    assert done["g-002"]["ok"] is True


def test_load_done_is_empty_when_nothing_recorded(tmp_path, monkeypatch):
    import run_agent_eval as h
    monkeypatch.setattr(h, "ROOT", tmp_path)
    assert load_done(tmp_path / "never_run.jsonl") == {}


def test_summarise_counts_resumed_and_fresh_records_alike(capsys):
    rows = [{"id": f"g-{i:03d}"} for i in range(1, 5)]
    recs = [
        {"id": "g-001", "kind": "answerable", "answered": True, "cites_gold": True,
         "faithful": True, "relevant": True, "gen_tokens": 100, "judge_tokens": 50},
        {"id": "g-002", "kind": "answerable", "answered": True, "cites_gold": True,
         "faithful": False, "relevant": True, "gen_tokens": 100, "judge_tokens": 50},
        {"id": "g-003", "kind": "answerable", "answered": False, "gen_tokens": 100},
        {"id": "g-004", "kind": "trap", "ok": True, "gen_tokens": 100},
    ]
    stats = summarise(Path("x.jsonl"), rows, recs, {"one-judge"}, None)
    assert stats["ans"] == 3 and stats["answered"] == 2
    assert stats["false_refusals"] == 1
    assert stats["judged"] == 2 and stats["faithful"] == 1
    assert stats["rel_judged"] == 2 and stats["relevant"] == 2
    assert stats["traps"] == 1 and stats["traps_refused"] == 1
    assert stats["tokens"] == 500


def test_summarise_warns_when_more_than_one_judge_ran(capsys):
    recs = [{"id": "g-001", "kind": "answerable", "answered": True,
             "faithful": True, "gen_tokens": 1}]
    summarise(Path("x.jsonl"), [{"id": "g-001"}], recs, {"judge-a", "judge-b"}, None)
    assert "more than one judge" in capsys.readouterr().out


def test_summarise_flags_an_incomplete_run(capsys):
    recs = [{"id": "g-001", "kind": "trap", "ok": True, "gen_tokens": 1}]
    summarise(Path("x.jsonl"), [{"id": "g-001"}, {"id": "g-002"}], recs, set(), "g-002")
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out and "--resume" in out
