"""Unit tests for loading, cleaning and the three chunking strategies."""

import re

from ingest.chunkers import chunk_fixed, chunk_recursive, chunk_structure, clean_text, doc_title


SAMPLE_MD = """# Test Scheme

> Source: https://example.gov.in (provenance — must not be indexed)
> Retrieved: 2026-08-27

Ministry: Ministry of Testing

## Benefits

A benefit of Rs. 5,000 per year is provided to every eligible family.

## Eligibility

All families with a test card are eligible under the scheme.
"""


def test_clean_strips_provenance_blockquotes():
    cleaned = clean_text(SAMPLE_MD)
    assert "Source:" not in cleaned
    assert "provenance" not in cleaned
    assert "Rs. 5,000" in cleaned          # real content survives


def test_doc_title_prefers_h1_with_filename_fallback():
    assert doc_title(SAMPLE_MD, "fallback") == "Test Scheme"
    assert doc_title("no heading here", "fallback") == "fallback"


def test_structure_chunks_carry_title_and_section_labels():
    chunks = chunk_structure(clean_text(SAMPLE_MD), title="Test Scheme")
    assert chunks, "expected at least one chunk"
    assert all(c.startswith("[Test Scheme") for c in chunks)
    joined = "\n".join(chunks)
    assert "Benefits]" in joined and "Eligibility]" in joined


def test_structure_chunks_never_start_mid_word():
    long_text = clean_text(SAMPLE_MD) * 20
    chunks = chunk_structure(long_text, title="Test Scheme")
    assert not any(re.match(r"^[a-z]", c) for c in chunks)


def test_fixed_and_recursive_split_long_documents():
    long_text = ("Paragraph about scheme benefits and eligibility rules. " * 8 + "\n\n") * 12
    assert len(chunk_fixed(long_text)) > 1
    assert len(chunk_recursive(long_text)) > 1


def test_recursive_overlap_starts_with_complete_words():
    # The overlap tail must never begin with a sliced fragment like
    # "ccountants" (Week 1's famous bug) — so every chunk's first word must
    # be a real word that occurs in the source text.
    long_text = ("Sentence with several ordinary words repeated here. " * 6 + "\n\n") * 10
    vocabulary = set(re.findall(r"[A-Za-z]+", long_text))
    for chunk in chunk_recursive(long_text):
        first = re.match(r"[A-Za-z]+", chunk)
        assert first is not None and first.group(0) in vocabulary
