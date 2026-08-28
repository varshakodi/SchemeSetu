"""Unit tests for the from-scratch BM25 — each test pins one property of the
formula, so a future refactor that breaks the math fails loudly."""

from naive.bm25 import BM25, tokenize


DOCS = [
    "NMMSS scholarship for meritorious students of economically weaker sections",
    "PMS-SC scholarship for scheduled caste students income ceiling",
    "crop insurance premium for farmers against natural disasters",
]


def test_tokenize_lowercases_and_splits():
    assert tokenize("PM-KISAN eKYC, Rs.6000!") == ["pm", "kisan", "ekyc", "rs", "6000"]


def test_exact_name_query_finds_the_named_doc():
    scores = BM25(DOCS).scores("NMMSS full form")
    assert scores.index(max(scores)) == 0


def test_rare_word_beats_common_word():
    bm25 = BM25(DOCS)
    # "scholarship" appears in two docs, "insurance" in one — the rarer
    # term must carry more weight (IDF).
    assert bm25.idf["insurance"] > bm25.idf["scholarship"]


def test_term_frequency_saturates():
    docs = [
        "insurance premium detail",
        "insurance " + "premium " * 10 + "detail",
    ]
    s1, s2 = (BM25(docs).scores("premium")[i] for i in (0, 1))
    assert s2 > s1            # more mentions -> higher score...
    assert s2 < 10 * s1       # ...but nowhere near linearly (k1 saturation)


def test_unknown_word_scores_zero_everywhere():
    assert BM25(DOCS).scores("zzzqx") == [0.0, 0.0, 0.0]
