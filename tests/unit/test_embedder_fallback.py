from __future__ import annotations

import numpy as np

from src.memory.embedder import Embedder


def _cosine(left: list[float], right: list[float]) -> float:
    return float(np.dot(np.array(left), np.array(right)))


def test_feature_hashing_fallback_is_deterministic(monkeypatch) -> None:
    embedder = Embedder()
    monkeypatch.setattr(embedder, "_load_model", lambda: None)

    assert embedder.encode("claim evidence graph") == embedder.encode("claim evidence graph")


def test_feature_hashing_preserves_lexical_similarity(monkeypatch) -> None:
    embedder = Embedder()
    monkeypatch.setattr(embedder, "_load_model", lambda: None)
    query = embedder.encode("claim evidence graph verification")
    related = embedder.encode("evidence graph for claim verification")
    unrelated = embedder.encode("cooking recipe with tomatoes")

    assert _cosine(query, related) > _cosine(query, unrelated)


def test_feature_hashing_handles_chinese_ngrams(monkeypatch) -> None:
    embedder = Embedder()
    monkeypatch.setattr(embedder, "_load_model", lambda: None)
    query = embedder.encode("证据缺口驱动补充检索")
    related = embedder.encode("系统根据证据缺口继续检索")
    unrelated = embedder.encode("今天适合制作番茄炒蛋")

    assert _cosine(query, related) > _cosine(query, unrelated)
