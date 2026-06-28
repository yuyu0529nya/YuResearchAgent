"""
tests/unit/test_dag.py
src/planner/dag.py 的单元测试：建图、依赖/后继、拓扑排序、环检测、并行分层。
纯逻辑，无重型依赖。
"""
import pytest

from src.planner.dag import DAG, DAGCycleError


def test_add_nodes_dedup_and_len():
    d = DAG()
    d.add_node("a")
    d.add_node("b")
    d.add_node("a")  # 重复静默忽略
    assert len(d) == 2
    assert d.has_node("a") and "b" in d


def test_add_edge_autocreates_nodes_and_relations():
    d = DAG()
    d.add_edge("a", "b")  # b 依赖 a
    assert d.has_node("a") and d.has_node("b")
    assert d.get_successors("a") == ["b"]
    assert d.get_dependencies("b") == ["a"]


def test_self_loop_raises():
    d = DAG()
    with pytest.raises(DAGCycleError):
        d.add_edge("x", "x")


def test_topological_sort_respects_order():
    d = DAG()
    d.add_edge("a", "b")
    d.add_edge("b", "c")
    order = d.topological_sort()
    assert order.index("a") < order.index("b") < order.index("c")


def test_topological_sort_detects_cycle():
    d = DAG()
    d.add_edge("a", "b")
    d.add_edge("b", "c")
    d.add_edge("c", "a")  # 成环
    with pytest.raises(DAGCycleError):
        d.topological_sort()


def test_parallel_groups_diamond():
    # a、b 为并行根；c 依赖 a 和 b；d 依赖 c
    d = DAG()
    for u, v in [("a", "c"), ("b", "c"), ("c", "d")]:
        d.add_edge(u, v)
    groups = d.get_parallel_groups()
    assert groups[0] == ["a", "b"]  # 同层并行，字典序稳定
    assert groups[1] == ["c"]
    assert groups[2] == ["d"]


def test_parallel_groups_all_independent_single_layer():
    d = DAG()
    for n in ["x", "y", "z"]:
        d.add_node(n)
    groups = d.get_parallel_groups()
    assert len(groups) == 1
    assert sorted(groups[0]) == ["x", "y", "z"]


def test_to_dict_shape():
    d = DAG()
    d.add_edge("a", "b")
    dd = d.to_dict()
    assert dd["nodes"] == ["a", "b"]
    assert dd["edges"]["a"] == ["b"]
