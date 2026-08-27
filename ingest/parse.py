"""SchemeSetu — PDF-to-text extraction. The dirtiest, most load-bearing step in RAG.

WHY PARSING MATTERS MORE THAN IT LOOKS: retrieval can only find what survived
extraction. A PDF is not text — it is a *drawing* of text: instructions like
"place these glyphs at these coordinates". Extraction reconstructs reading
order from coordinates, and that reconstruction mangles things: multi-column
layouts interleave, tables collapse into word soup, headers/footers repeat
into the middle of sentences, and scanned pages contain no text at all (they
are photographs — only OCR, optical character recognition, can read those).
Garbage in here means garbage retrieved later, no matter how good the
embeddings are. So this script *reports* extraction quality per page instead
of failing silently.

Usage:
    python ingest/parse.py          # convert every data/raw/*.pdf to a sibling .txt

The .txt outputs are what naive/rag.py indexes (it reads .md and .txt only,
so unparsed PDFs are invisible to retrieval until this script runs).
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

from pypdf import PdfReader

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# A page whose extracted text is shorter than this is suspicious — probably a
# scanned image, a diagram, or a parsing failure. We flag it, not hide it.
THIN_PAGE_CHARS = 80


def parse_pdf(pdf_path: Path) -> tuple[str, int, list[int]]:
    """Extract text page by page; return (text, page_count, thin_page_numbers)."""
    reader = PdfReader(pdf_path)
    pages, thin = [], []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if len(text.strip()) < THIN_PAGE_CHARS:
            thin.append(i)
        pages.append(text)
    joined = "\n\n".join(pages)
    joined = re.sub(r"\n{3,}", "\n\n", joined)  # collapse runs of blank lines
    return joined, len(pages), thin


def main() -> None:
    pdfs = sorted(RAW.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RAW} — nothing to do.")
        return

    for pdf in pdfs:
        text, n_pages, thin = parse_pdf(pdf)
        out = pdf.with_suffix(".txt")
        header = (f"# Extracted from {pdf.name} ({n_pages} pages) on "
                  f"{datetime.date.today().isoformat()} by ingest/parse.py\n"
                  f"# Source and provenance: see data/registry.csv\n\n")
        out.write_text(header + text, encoding="utf-8")

        status = "OK" if not thin else f"WARNING: {len(thin)} thin/empty pages {thin} (scanned? tables?)"
        print(f"{pdf.name}: {n_pages} pages -> {out.name} ({len(text)} chars) [{status}]")


if __name__ == "__main__":
    main()
