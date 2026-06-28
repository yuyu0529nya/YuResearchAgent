"""
tests/unit/test_extractive.py
src/compressor/extractive.py 纯 TextRank 函数的单元测试。

按文件路径加载（embedder 已改懒加载，故无需 sentence-transformers）。
只覆盖与 embedding 无关的纯逻辑：分句、相似度矩阵、PageRank 迭代。
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "compressor", "extractive.py",
)
_spec = importlib.util.spec_from_file_location("extractive_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
E = _mod


def test_tokenize_filters_short_sentences():
    sents = E._tokenize_sentences("这是一个足够长的句子用于测试分句功能。另一个同样足够长的句子在这里。")
    assert len(sents) == 2


def test_tokenize_mixed_punctuation():
    sents = E._tokenize_sentences("First long enough sentence here! Second long enough sentence too?")
    assert len(sents) == 2


def test_cosine_matrix_identity_and_orthogonal():
    v = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    sim = E._cosine_similarity_matrix(v)
    assert np.allclose(np.diag(sim), 1.0)
    assert sim[0, 1] == pytest.approx(1.0)   # 相同向量
    assert sim[0, 2] == pytest.approx(0.0)   # 正交向量


def test_cosine_matrix_handles_zero_vector():
    # 零向量不应导致除零崩溃
    v = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    sim = E._cosine_similarity_matrix(v)
    assert sim.shape == (2, 2)
    assert np.isfinite(sim).all()


def test_textrank_scores_normalized_and_positive():
    sim = np.array([[1, 0.8, 0.0], [0.8, 1, 0.0], [0.0, 0.0, 1]])
    sc = E._textrank_scores(sim)
    assert np.all(sc >= 0)
    assert sc.sum() == pytest.approx(1.0, abs=0.05)


def test_textrank_central_sentence_ranks_highest():
    # 句 0 与其余三句都高度相似（中心节点），应得分最高
    sim = np.array([
        [1, 0.8, 0.8, 0.8],
        [0.8, 1, 0.2, 0.2],
        [0.8, 0.2, 1, 0.2],
        [0.8, 0.2, 0.2, 1],
    ])
    sc = E._textrank_scores(sim)
    assert int(np.argmax(sc)) == 0


def test_textrank_empty_matrix():
    assert E._textrank_scores(np.zeros((0, 0))).size == 0
