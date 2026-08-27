"""SchemeSetu — document loading, cleaning, and three chunking strategies.

WEEK 2's QUESTION: does *how* we cut documents into chunks measurably change
retrieval quality? Chunking decides what a chunk "is", and therefore what an
embedding can represent and what retrieval can find. We compare three
strategies under identical conditions (same corpus, same embedder, same eval
questions — vary ONE thing at a time) and let the numbers pick the default.

The three contestants:

  fixed      Week 1's baseline: pack paragraphs to ~800 chars, overlap by
             copying the previous chunk's last 150 *characters* — which is how
             we ended up with a chunk starting mid-word ("ccountants...").

  recursive  Try to split on the biggest natural boundary first (blank line),
             and only fall back to smaller ones (line, sentence, word) when a
             piece is still too big. Overlap snaps to word boundaries. This is
             a from-scratch version of what LangChain calls
             RecursiveCharacterTextSplitter — when we adopt the framework
             later, you'll know exactly what it does.

  structure  Use the document's own skeleton: split at markdown headings, keep
             sections whole where possible, and prefix every chunk with
             "[Document title › Section heading]". A chunk that says where it
             comes from carries its own context — the embedding of
             "[NMMSS › Details] income not more than ₹3.5 lakh" knows it is
             about NMMSS, even though the sentence alone doesn't say so.

Cleaning: our corpus files begin with provenance blockquotes ("> Source: ...")
and extraction headers. Useful for humans, poison for retrieval — Week 1's
eval showed a provenance header outranking real content. `clean_text()`
strips that boilerplate before chunking, uniformly for every strategy.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC_DIRS = [ROOT / "data" / "samples", ROOT / "data" / "raw"]

TARGET_CHARS = 800     # same target for every strategy — we compare *strategy*,
OVERLAP_CHARS = 150    # not size. (Size is a separate experiment for later.)


# --- Loading & cleaning -------------------------------------------------------

def clean_text(text: str) -> str:
    """Strip provenance boilerplate that must not be indexed.

    Removes blockquote lines ("> Source: ...") and extraction headers. Our
    corpus uses blockquotes only for provenance, so dropping them all is safe
    here — revisit if real quoted content ever enters the corpus.
    """
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(">"):
            continue
        if s.startswith("# Extracted from") or s.startswith("# Source and provenance"):
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def doc_title(text: str, fallback: str) -> str:
    """First markdown H1 if present, else the filename — used as chunk context."""
    for line in text.splitlines():
        m = re.match(r"^# (.+)$", line.strip())
        if m:
            return m.group(1).strip()
    return fallback


def load_documents(clean: bool = True) -> list[tuple[str, str, str]]:
    """Return (doc_id, title, text) for every .md/.txt in the corpus folders."""
    docs = []
    for d in DOC_DIRS:
        if not d.exists():
            continue
        for path in sorted(d.glob("*")):
            if path.suffix.lower() in {".md", ".txt"}:
                raw = path.read_text(encoding="utf-8")
                title = doc_title(raw, fallback=path.stem)
                docs.append((path.name, title, clean_text(raw) if clean else raw))
    return docs


# --- Strategy 1: fixed (the Week 1 baseline, unchanged) -----------------------

def chunk_fixed(text: str, title: str = "") -> list[str]:
    """Pack paragraphs to ~TARGET_CHARS; overlap = raw character tail.

    Kept byte-for-byte equivalent to Week 1 so it remains an honest baseline —
    including its known flaw: the character-tail overlap can slice words.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) > TARGET_CHARS:
            chunks.append(current.strip())
            current = current[-OVERLAP_CHARS:]
        current += "\n\n" + para
    if current.strip():
        chunks.append(current.strip())
    return chunks


# --- Strategy 2: recursive splitting ------------------------------------------

_SEPS = ["\n\n", "\n", ". ", " "]


def _pack(units: list[str], sep: str, target: int) -> list[str]:
    """Greedily merge small units back together, up to the target size."""
    pieces, cur = [], ""
    for u in units:
        u = u.strip("\n")
        if not u.strip():
            continue
        cand = f"{cur}{sep}{u}" if cur else u
        if cur and len(cand) > target:
            pieces.append(cur)
            cur = u
        else:
            cur = cand
    if cur:
        pieces.append(cur)
    return pieces


def _split_recursive(text: str, sep_i: int = 0) -> list[str]:
    if len(text) <= TARGET_CHARS:
        return [text]
    if sep_i >= len(_SEPS):  # no separators left: hard cut (last resort)
        return [text[i:i + TARGET_CHARS] for i in range(0, len(text), TARGET_CHARS)]
    units = text.split(_SEPS[sep_i])
    if len(units) == 1:
        return _split_recursive(text, sep_i + 1)
    out = []
    for piece in _pack(units, _SEPS[sep_i], TARGET_CHARS):
        out.extend(_split_recursive(piece, sep_i + 1) if len(piece) > TARGET_CHARS else [piece])
    return out


def _word_tail(text: str, n: int) -> str:
    """Last ~n characters, trimmed forward to the next word boundary."""
    tail = text[-n:]
    cut = tail.find(" ")
    return tail[cut + 1:].strip() if cut != -1 else tail.strip()


def chunk_recursive(text: str, title: str = "") -> list[str]:
    """Split on the largest natural boundary available; word-boundary overlap."""
    pieces = [p.strip() for p in _split_recursive(text.strip()) if p.strip()]
    chunks = []
    for i, piece in enumerate(pieces):
        if i > 0:
            piece = _word_tail(pieces[i - 1], OVERLAP_CHARS) + "\n" + piece
        chunks.append(piece)
    return chunks


# --- Strategy 3: structure-aware ----------------------------------------------

_HEADING = re.compile(r"^(#{1,3})\s+(.+)$")


def chunk_structure(text: str, title: str = "") -> list[str]:
    """Split at markdown headings; label every chunk with its origin.

    Sections stay whole when they fit; oversized sections are split by the
    recursive strategy, each part keeping the label; small neighbouring
    sections are packed together. Documents without headings (e.g. parsed
    PDFs) degrade gracefully to recursive splitting with a title label.
    """
    sections: list[tuple[str, str]] = []
    heading, buf = "", []
    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            if "\n".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if "\n".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))

    labelled: list[str] = []
    for heading, body in sections:
        label = f"[{title} › {heading}]" if heading and heading != title else f"[{title}]"
        if len(body) + len(label) <= TARGET_CHARS * 1.4:  # small tolerance to keep sections whole
            labelled.append(f"{label}\n{body}")
        else:
            labelled.extend(f"{label}\n{part}" for part in chunk_recursive(body))
    return _pack(labelled, "\n\n", TARGET_CHARS)


CHUNKERS = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "structure": chunk_structure,
}
