"""Review helper for the golden set.

Prints each candidate question next to the passage in its gold document that
best supports the draft answer, so a human can verify ground truth quickly.

This is a LABELLING tool, not an eval. It deliberately does not use the
retrieval system: labels must be checked independently of the thing they
will be used to judge, otherwise a retrieval bug can quietly rewrite the
answer key. Overlap here is plain word matching, nothing learned.

Usage:
    python evals/review_golden.py                  # everything
    python evals/review_golden.py --type factual   # one slice
    python evals/review_golden.py --weak           # only suspicious labels
    python evals/review_golden.py --file evals/golden_set.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

CANDIDATES = "evals/golden_candidates.jsonl"
# Below this word-overlap fraction, the gold document probably does not
# actually contain the answer -> the label needs a human look.
WEAK_SUPPORT = 0.34
STOP = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "is", "are", "on",
    "at", "by", "per", "with", "from", "as", "be", "it", "that", "this", "not",
    "who", "what", "how", "much", "can", "do", "does", "i", "my", "you", "if",
}


def corpus_paths() -> dict[str, str]:
    return {
        os.path.basename(p): p
        for p in glob.glob("data/**/*.*", recursive=True)
        if p.endswith((".md", ".txt"))
    }


def words(text: str) -> set[str]:
    """Content words, with rupee amounts normalised so 1,20,000 == 120000."""
    text = text.lower().replace(",", "")
    return {w for w in re.findall(r"[\wऀ-ॿ]+", text) if w not in STOP}


def passages(text: str) -> list[str]:
    """Split a document into paragraph-ish units.

    These documents write "## Benefits" immediately above the paragraph with
    no blank line, so a heading arrives glued to the text it introduces.
    Strip the marker instead of discarding the block -- dropping anything that
    starts with "#" throws away most of the corpus.
    """
    out: list[str] = []
    for part in re.split(r"\n\s*\n", text):
        p = " ".join(part.split())
        p = re.sub(r"^#+\s*", "", p)
        if len(p) > 40:
            out.append(p)
    return out


def is_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))


def crosslingual_tokens(text: str) -> set[str]:
    """Language-neutral anchors: figures and Latin acronyms.

    A Hindi answer shares no words with an English source document, so plain
    overlap always reads 0% and flags every Hindi row. The facts that must
    match across languages are the numbers and the scheme acronyms, so those
    are what we check.
    """
    text = text.replace(",", "")
    return {t.lower() for t in re.findall(r"\d{2,}|[A-Z]{2,}", text)}


def best_support(answer: str, docs: list[str], paths: dict[str, str]):
    """How well the gold documents support `answer`.

    Returns (doc, passage, support, mode). `support` is measured against the
    UNION of the gold documents: a synthesis answer is deliberately spread
    across several sources, so scoring it against one passage would punish it
    for being what it is. The passage returned is the single best one, for
    reading.
    """
    mode = "cross-lingual" if is_devanagari(answer) else "words"
    pick = crosslingual_tokens if mode == "cross-lingual" else words

    target = pick(answer)
    if not target:
        # A Hindi answer with no figures and no acronyms gives the checker
        # nothing language-neutral to match on. Say so; do not report 0%,
        # which reads as "unsupported" when it means "unverifiable here".
        return docs[0], "", None, "no anchors"
    union: set[str] = set()
    best = (docs[0], "", 0.0)
    for doc in docs:
        raw = open(paths[doc], encoding="utf-8").read()
        for p in passages(raw):
            union |= pick(p)
            hit = len(target & pick(p)) / max(1, len(target))
            if hit > best[2]:
                best = (doc, p, hit)
    support = len(target & union) / max(1, len(target))
    return best[0], best[1], support, mode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=CANDIDATES)
    ap.add_argument("--type", help="factual | synthesis | hindi | trap | ambiguous")
    ap.add_argument("--weak", action="store_true", help="only weakly supported labels")
    args = ap.parse_args()

    paths = corpus_paths()
    rows = [json.loads(line) for line in open(args.file, encoding="utf-8")]
    if args.type:
        rows = [r for r in rows if r["type"] == args.type]

    shown = weak = blind = 0
    for r in rows:
        if not r["gold_doc_ids"]:
            if args.weak:
                continue
            shown += 1
            print(f"\n[{r['id']}] {r['type'].upper()} · {r['language']}")
            print(f"  Q: {r['question']}")
            print(f"  Expected: {r['answer']}")
            print("  (no gold document — the system should decline or ask)")
            continue

        doc, passage, overlap, mode = best_support(r["answer"], r["gold_doc_ids"], paths)
        unverifiable = overlap is None
        is_weak = (not unverifiable) and overlap < WEAK_SUPPORT
        weak += is_weak
        blind += unverifiable
        if args.weak and not (is_weak or unverifiable):
            continue
        shown += 1
        flag = "  <-- CHECK THIS" if is_weak else ""
        score = "not auto-checkable" if unverifiable else f"support {overlap:.0%}"
        print(f"\n[{r['id']}] {r['type'].upper()} · {r['language']} · {score} ({mode}){flag}")
        print(f"  Q: {r['question']}")
        print(f"  Draft answer: {r['answer']}")
        print(f"  Gold docs: {', '.join(r['gold_doc_ids'])}")
        print(f"  Best supporting passage ({doc}):")
        print(f"    {passage[:420]}{'...' if len(passage) > 420 else ''}")

    print(
        f"\n--- {shown} shown · {weak} weakly supported · "
        f"{blind} not auto-checkable (verify by hand) · {len(rows)} total ---"
    )


if __name__ == "__main__":
    main()
