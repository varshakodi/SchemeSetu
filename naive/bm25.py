"""BM25 from scratch — the keyword scorer that ran search engines for decades.

WHY DO WE NEED THIS WHEN WE HAVE EMBEDDINGS? They fail differently:

  Embeddings understand MEANING but blur NAMES. "Am I eligible?" matches
  "Who qualifies?" — brilliant. But "NMMSS" and "PMS-SC" are both just
  ~"scholarship acronym" to an embedding; the exact letters barely matter.

  Keyword search matches LETTERS but not meaning. "NMMSS" finds exactly the
  chunks containing NMMSS — perfect. But "money for my new baby" finds
  nothing about "maternity benefit", because no words overlap.

A production retriever runs BOTH and merges the results ("hybrid retrieval").
This file is the keyword half, implemented from scratch so you can explain
every term of the formula in an interview.

THE IDEA, BUILT UP IN THREE STEPS:

  1. TF (term frequency): a chunk mentioning "premium" three times is more
     about premiums than one mentioning it once.
  2. IDF (inverse document frequency): rare words carry more signal. Every
     chunk contains "scheme", so matching it means nothing; few contain
     "subvention", so matching it means a lot. IDF weighs each word by how
     rare it is across the corpus.
  3. BM25's two refinements over plain TF-IDF:
       - saturation (k1): the 10th repetition of a word adds far less than
         the 2nd — TF's contribution flattens out instead of growing forever.
       - length normalisation (b): long chunks mention everything a little;
         a match inside a short chunk is stronger evidence than the same
         match inside a rambling one.

The scoring formula per query word w and chunk d:

    score(w, d) = IDF(w) * TF(w,d) * (k1 + 1)
                  ------------------------------------------------
                  TF(w,d) + k1 * (1 - b + b * len(d) / avg_len)

k1=1.5 and b=0.75 are the standard defaults from the literature.
(The production equivalent is the `rank_bm25` package — same algorithm.)
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric words — deliberately simple.

    Real systems add stemming ("instalments" -> "instalment") and stopword
    removal; we skip both for now and let the eval tell us if it matters.
    """
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.tf = [Counter(tokens) for tokens in self.doc_tokens]

        # document frequency: in how many chunks does each word appear?
        df: Counter = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n = len(docs)
        # +0.5 smoothing and +1 inside the log keep IDF positive and finite
        # even for words that appear in almost every chunk.
        self.idf = {w: math.log((n - dfw + 0.5) / (dfw + 0.5) + 1) for w, dfw in df.items()}

    def scores(self, query: str) -> list[float]:
        """BM25 score of the query against every chunk (higher = better)."""
        out = [0.0] * len(self.doc_tokens)
        for w in tokenize(query):
            idf = self.idf.get(w)
            if idf is None:            # word never seen in the corpus
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out
