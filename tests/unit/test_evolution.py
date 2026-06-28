"""
tests/unit/test_evolution.py
M6 自进化引擎「可测组件」单元测试 —— 无需 GPU / 训练 / 网络。

目的：用代码证明 evolution 不是空壳，其算法是真实现：
  - TrajectoryCollector：轨迹收集 + VERL 训练格式转换
  - Judge：GRPO 奖励塑形 [−1,1] + 效率分 sigmoid（防 reward hacking）
  - Proposer：倒 U 型难度权重 + 余弦多样性 + 兜底出题
  - SymbolicLearner：prompt 版本栈 / 性能回滚 / 磁盘持久化（免梯度的自进化）
  - ExperienceMemory：SQLite 经验库增查

注：GRPO「梯度更新」那一步本身是接 veRL 的占位（需 GPU+本地可训模型），
本测试覆盖的是其外围「训练数据管线 + 奖励 + 免梯度符号学习」，这些是真能跑的。
"""
import math

from src.evolution.collector import TrajectoryCollector
from src.evolution.experience_memory import ExperienceMemory
from src.evolution.judge import DIMENSION_WEIGHTS, Judge
from src.evolution.proposer import Proposer
from src.evolution.symbolic_learning import SymbolicLearner
from src.orchestrator.schemas import ResearchReport


def _report() -> ResearchReport:
    return ResearchReport(query="q", content="报告内容", sources=[{"title": "t", "url": "u"}])


# --------------------------- TrajectoryCollector ---------------------------
def test_collector_collect_fields():
    d = TrajectoryCollector().collect("q", _report(), [{"role": "user", "content": "x"}])
    assert d["query"] == "q"
    assert d["trajectory_length"] == 1
    assert d["source_count"] == 1
    assert d["content_length"] == len("报告内容")


def test_collector_to_verl_format():
    c = TrajectoryCollector(system_prompt="SYS")
    verl = c.to_verl_format(c.collect("研究问题", _report(), []))
    assert verl["prompt"][0] == {"role": "system", "content": "SYS"}
    assert verl["prompt"][-1] == {"role": "user", "content": "研究问题"}
    assert verl["response"] == "报告内容"
    assert verl["metadata"]["source_count"] == 1


def test_collector_batch_to_verl():
    c = TrajectoryCollector()
    out = c.batch_to_verl([c.collect(f"q{i}", _report(), []) for i in range(3)])
    assert len(out) == 3 and all("prompt" in x for x in out)


# --------------------------- Judge：奖励塑形 / 效率分 ---------------------------
def test_dimension_weights_sum_to_one():
    assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9


def test_shape_reward_bounds_and_midpoint():
    j = Judge(policy=None)
    assert j.shape_reward({k: 10.0 for k in DIMENSION_WEIGHTS}) == 1.0
    assert j.shape_reward({k: 0.0 for k in DIMENSION_WEIGHTS}) == -1.0
    assert abs(j.shape_reward({k: 0.5 for k in DIMENSION_WEIGHTS}) - 0.0) < 1e-9


def test_shape_reward_clips_out_of_range():
    j = Judge(policy=None)
    assert j.shape_reward({k: 100.0 for k in DIMENSION_WEIGHTS}) == 1.0
    assert j.shape_reward({k: -50.0 for k in DIMENSION_WEIGHTS}) == -1.0


def test_efficiency_score_sigmoid_monotonic():
    j = Judge(policy=None, efficiency_optimal=5, efficiency_scale=3.0)
    assert abs(j._compute_efficiency_score(5) - 5.0) < 1e-9  # optimal → 5.0
    assert j._compute_efficiency_score(0) > j._compute_efficiency_score(5)
    assert j._compute_efficiency_score(5) > j._compute_efficiency_score(20)
    assert 0.0 < j._compute_efficiency_score(50) < 10.0


# --------------------------- Proposer：纯方法 ---------------------------
def test_proposer_difficulty_weights_normalized():
    w = Proposer(policy=None).get_difficulty_weights()
    assert set(w) == {"L1", "L2", "L3"}
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_proposer_cosine_similarity():
    cs = Proposer._cosine_similarity
    assert abs(cs([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-9
    assert abs(cs([1, 0], [0, 1]) - 0.0) < 1e-9
    assert cs([1, 2, 3], [1, 2]) == 0.0  # 长度不等
    assert cs([0, 0], [1, 1]) == 0.0  # 零向量


def test_proposer_fallback_question_contains_domain():
    p = Proposer(policy=None)
    for lvl in ["L1", "L2", "L3"]:
        q = p._fallback_question(lvl, ["医疗"])
        assert isinstance(q, str) and "医疗" in q


def test_proposer_is_diverse_empty_cache():
    assert Proposer(policy=None)._is_diverse("任意新问题") is True


# --------------------------- SymbolicLearner：版本 / 回滚 / 持久化 ---------------------------
def test_symbolic_save_and_rollback():
    s = SymbolicLearner(policy=None)
    s._save_version({"sys": "v1"})
    s._save_version({"sys": "v2"})
    assert s._rollback_one() == {"sys": "v1"}


def test_symbolic_rollback_needs_two_versions():
    s = SymbolicLearner(policy=None)
    s._save_version({"sys": "v1"})
    assert s._rollback_one() is None


def test_symbolic_rollback_if_needed_on_perf_drop():
    s = SymbolicLearner(policy=None, rollback_threshold=0.05)
    s._save_version({"sys": "good"})
    s._save_version({"sys": "bad"})
    s.rollback_if_needed({"sys": "bad"}, {"avg_score": 8.0})  # 建立基线
    out = s.rollback_if_needed({"sys": "bad"}, {"avg_score": 5.0})  # 大幅下降 → 回滚
    assert out == {"sys": "good"}


def test_symbolic_disk_roundtrip(tmp_path):
    s = SymbolicLearner(policy=None)
    s._save_version({"sys": "a"})
    s._save_version({"sys": "b"})
    d = str(tmp_path / "versions")
    s.save_versions_to_disk(d)
    s2 = SymbolicLearner(policy=None)
    s2.load_versions_from_disk(d)
    assert s2._prompt_versions == [{"sys": "a"}, {"sys": "b"}]


# --------------------------- ExperienceMemory：SQLite 增查 ---------------------------
def test_experience_memory_add_and_retrieve(tmp_path):
    m = ExperienceMemory(db_path=str(tmp_path / "exp.db"))
    rid = m.add(trajectory=[{"step": "search"}], success=True, score=8.0, strategy_summary="先检索后合成")
    assert isinstance(rid, int) and rid >= 1
    results = m.retrieve("检索策略", top_k=3)
    assert isinstance(results, list) and len(results) >= 1
    stats = m.get_stats()
    assert isinstance(stats, dict) and stats
