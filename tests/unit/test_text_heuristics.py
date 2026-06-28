"""
tests/unit/test_text_heuristics.py
src/memory/text_heuristics.py 的单元测试：矛盾检测启发式（反义词对 + 否定不对称）。

注意：测试固定的是**当前真实行为**，包括已知局限（中文无空格分词导致否定式
Jaccard 失效）。如后续改进启发式，应同步更新这些测试。
"""
from src.memory.text_heuristics import is_semantically_opposite as opp


def test_antonym_pair_chinese_increase_decrease():
    assert opp("销量大幅增加", "销量大幅减少") is True


def test_antonym_pair_english_good_bad():
    assert opp("the result is good", "the result is bad") is True


def test_support_versus_oppose():
    assert opp("大家都支持这个方案", "大家都反对这个方案") is True


def test_unrelated_claims_not_opposite():
    assert opp("今天天气很好", "我喜欢吃苹果") is False


def test_same_direction_not_opposite():
    assert opp("销量增加了", "成本增加了") is False


def test_english_negation_asymmetry_detected():
    # 英文以空格分词，否定不对称 + 去否定后高度相似 → 判定对立
    assert opp("the plan is good", "the plan is not good") is True


def test_chinese_negation_is_known_limitation():
    # 已知局限：中文无空格分词，否定式 Jaccard 相似度失效 → 当前返回 False。
    # 固定此行为以提醒：若未来引入中文分词改进，需更新本断言。
    assert opp("项目成功了", "项目没有成功") is False
