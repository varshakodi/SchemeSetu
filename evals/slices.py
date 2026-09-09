"""How an eval row is graded.

Both harnesses used to ask "does this row have gold documents?" and treat
everything without them as a trap. That was fine while traps were the only
unanswerable rows. The golden set also has AMBIGUOUS rows -- underspecified
questions like "Am I eligible?" -- which have no gold documents either, and
which must NOT be scored as traps: refusing them is not the goal, and
counting them in the trap rate silently changes what that number means.

So the declared type decides, with the old gold-document test kept as a
fallback for sets written before types existed.
"""
from __future__ import annotations

ANSWERABLE = "answerable"
TRAP = "trap"
AMBIGUOUS = "ambiguous"


def slice_of(row: dict) -> str:
    kind = row.get("type")
    if kind in (TRAP, AMBIGUOUS):
        return kind
    return ANSWERABLE if row.get("gold_doc_ids") else TRAP


def split(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (answerable, traps, ambiguous)."""
    return (
        [r for r in rows if slice_of(r) == ANSWERABLE],
        [r for r in rows if slice_of(r) == TRAP],
        [r for r in rows if slice_of(r) == AMBIGUOUS],
    )
