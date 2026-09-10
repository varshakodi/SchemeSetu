"""Re-indexing must not silently replace a full index with a partial one.

The repository ships a prebuilt index (132 chunks over 23 documents) because
the corpus is not committed. A fresh clone therefore has only the 3 sample
documents on disk. Following the old README, `python naive/rag.py index`
rebuilt from those 3 and wrote a 10-chunk index over the shipped one --
no error, no warning, just a quietly crippled system.

The guard refuses a rebuild that would at least halve the index unless
--force is passed.
"""

import json

import pytest

from naive import rag


def _fake_corpus(monkeypatch, tmp_path, n_docs):
    chunks = tmp_path / "chunks.json"
    monkeypatch.setattr(rag, "CHUNKS_FILE", chunks)
    monkeypatch.setattr(rag, "INDEX_FILE", tmp_path / "index.npz")
    # Pretend the shipped index already holds a full corpus.
    chunks.write_text(json.dumps([{"doc_id": "d.md", "chunk_id": f"d.md#{i}",
                                   "text": "x"} for i in range(132)]))
    monkeypatch.setattr(rag, "load_documents",
                        lambda: [(f"d{i}.md", f"Doc {i}", "para one\n\npara two")
                                 for i in range(n_docs)])
    return chunks


def test_refuses_to_shrink_a_full_index(monkeypatch, tmp_path):
    _fake_corpus(monkeypatch, tmp_path, n_docs=1)
    with pytest.raises(SystemExit) as exc:
        rag.build_index()
    message = str(exc.value)
    assert "Refusing to overwrite" in message
    assert "--force" in message, "the message must say how to proceed deliberately"


def test_force_allows_the_rebuild(monkeypatch, tmp_path):
    """--force reaches embedding; that it then needs a model is not our concern."""
    _fake_corpus(monkeypatch, tmp_path, n_docs=1)
    monkeypatch.setattr(rag, "get_embedder", lambda: (_ for _ in ()).throw(
        RuntimeError("reached embedding")))
    with pytest.raises(RuntimeError, match="reached embedding"):
        rag.build_index(force=True)


def test_similar_sized_rebuild_is_not_blocked(monkeypatch, tmp_path):
    """A normal re-index after editing one document must not trip the guard."""
    _fake_corpus(monkeypatch, tmp_path, n_docs=70)   # ~140 chunks vs 132 existing
    monkeypatch.setattr(rag, "get_embedder", lambda: (_ for _ in ()).throw(
        RuntimeError("reached embedding")))
    with pytest.raises(RuntimeError, match="reached embedding"):
        rag.build_index()
