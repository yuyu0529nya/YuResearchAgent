"""
src/memory/text_heuristics.py
================================================================================
矛盾检测的纯文本启发式（无重型依赖，可独立单测）。

从 ``memory_store`` 抽出，供 M4 共享记忆的矛盾检测使用：在两条 claim 的
embedding 相似度落在"疑似矛盾"区间（0.65 < cosine < 0.92）时，再用本模块
的 :func:`is_semantically_opposite` 做语义对立判定，降低误报。

不依赖 numpy / sentence-transformers，因此可在无嵌入环境下被单元测试覆盖。
================================================================================
"""
from __future__ import annotations

__all__ = ["is_semantically_opposite", "NEGATION_WORDS", "ANTONYM_PAIRS"]


# 常见否定词（中英）
NEGATION_WORDS: set[str] = {"不", "没", "无", "非", "未", "否", "not", "no", "never", "without"}

# 反义词对：(正向词集合, 反向词集合)
ANTONYM_PAIRS: list[tuple[set[str], set[str]]] = [
    ({"增加", "上升", "增长", "提高", "扩大", "increase", "rise", "grow"},
     {"减少", "下降", "降低", "缩减", "收缩", "decrease", "fall", "drop"}),
    ({"好", "优", "强", "positive", "good", "strong"},
     {"坏", "劣", "弱", "negative", "bad", "weak"}),
    ({"支持", "赞成", "agree", "support"},
     {"反对", "reject", "oppose", "disagree"}),
    ({"成功", "success"}, {"失败", "failure"}),
    ({"高", "high"}, {"低", "low"}),
    ({"大", "big", "large"}, {"小", "small", "tiny"}),
]


def is_semantically_opposite(claim_a: str, claim_b: str) -> bool:
    """简单启发式判断两个 claim 是否语义对立。

    策略：
        1. 一方含否定词、另一方不含，且去除否定词后 Jaccard 相似度 > 0.5
           （结构相似但极性相反，如"X 上升" vs "X 不上升"）
        2. 一方含某正向词、另一方含其反义词（如"增加" vs "减少"）

    Args:
        claim_a: 第一条陈述。
        claim_b: 第二条陈述。

    Returns:
        判定为语义对立返回 True，否则 False。
    """
    ca = claim_a.lower()
    cb = claim_b.lower()

    # 否定词不对称检查
    a_has_neg = any(w in ca for w in NEGATION_WORDS)
    b_has_neg = any(w in cb for w in NEGATION_WORDS)
    if a_has_neg != b_has_neg:
        def _strip_neg(text: str) -> set[str]:
            words = set(text.split())
            for w in NEGATION_WORDS:
                words.discard(w)
            return words
        inter = len(_strip_neg(ca) & _strip_neg(cb))
        union = len(_strip_neg(ca) | _strip_neg(cb))
        if union > 0 and inter / union > 0.5:
            return True

    # 反义词对检查
    for pos_set, neg_set in ANTONYM_PAIRS:
        a_pos = any(w in ca for w in pos_set)
        a_neg = any(w in ca for w in neg_set)
        b_pos = any(w in cb for w in pos_set)
        b_neg = any(w in cb for w in neg_set)
        if (a_pos and b_neg) or (a_neg and b_pos):
            return True

    return False
