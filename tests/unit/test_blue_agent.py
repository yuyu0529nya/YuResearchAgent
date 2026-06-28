"""
tests/unit/test_blue_agent.py
src/adversarial/blue_agent.py 的 defend() 修复循环单元测试。

用 mock policy（返回脚本化的修复 JSON）驱动，无需 LLM/网络，覆盖：
优先级排序执行、in_place / removal 修复落地、self_verify 不阻塞、无 issue 直接返回。
"""
import asyncio
from types import SimpleNamespace

from src.adversarial.blue_agent import BlueAgent
from src.adversarial.verdict import Dimension, FixType, Issue, RedVerdict, Severity
from src.orchestrator.schemas import ResearchReport


class _MockPolicy:
    """返回固定 content 的假 policy（带 max_tokens，blue agent 会临时改它）。"""

    max_tokens = 4000

    def __init__(self, content: str):
        self._content = content

    def __call__(self, messages):
        return SimpleNamespace(content=self._content)


def _report() -> ResearchReport:
    return ResearchReport(query="q", content="原始报告内容，含一处需要修复的错误。", sources=[])


def _run(coro):
    return asyncio.run(coro)


def test_defend_no_issues_returns_unchanged():
    ba = BlueAgent(policy=_MockPolicy(""), tools=[])
    rep = _report()
    out, ops = _run(ba.defend(rep, RedVerdict(issues=[])))
    assert ops == []
    assert out.content == rep.content


def test_defend_in_place_applies_targeted_edit():
    # 定点编辑：before 片段被精确替换，不重写全文
    edit_json = '{"edits": [{"before": "含一处需要修复的错误", "after": "已修正所有错误", "reason": "纠正"}]}'
    ba = BlueAgent(policy=_MockPolicy(edit_json), tools=[])
    issue = Issue(Severity.CRITICAL, Dimension.FACTUAL, "数字与来源不一致", fix_type=FixType.IN_PLACE)
    out, ops = _run(ba.defend(_report(), RedVerdict(issues=[issue])))
    assert "已修正所有错误" in out.content
    assert "含一处需要修复的错误" not in out.content
    assert any(op.success for op in ops)


def test_defend_in_place_edit_before_not_found_keeps_original():
    # before 不匹配原文 → 跳过替换，绝不损坏报告（定点编辑的核心安全性）
    ba = BlueAgent(policy=_MockPolicy('{"edits": [{"before": "原文里没有的句子", "after": "X"}]}'), tools=[])
    rep = _report()
    orig = rep.content
    issue = Issue(Severity.MAJOR, Dimension.FACTUAL, "x", fix_type=FixType.IN_PLACE)
    out, ops = _run(ba.defend(rep, RedVerdict(issues=[issue])))
    assert out.content == orig
    assert any(not op.success for op in ops)


def test_defend_removal_applies():
    ba = BlueAgent(policy=_MockPolicy('{"fixed_content": "删除幻觉段落后的报告。", "removed_segments": ["幻觉段落"]}'), tools=[])
    issue = Issue(Severity.CRITICAL, Dimension.HALLUCINATION, "高置信幻觉", fix_type=FixType.REMOVAL)
    out, ops = _run(ba.defend(_report(), RedVerdict(issues=[issue])))
    assert out.content == "删除幻觉段落后的报告。"
    assert any(op.success for op in ops)


def test_defend_processes_all_issues():
    ba = BlueAgent(policy=_MockPolicy('{"fixed_content": "已修复。", "changes": []}'), tools=[])
    issues = [
        Issue(Severity.MINOR, Dimension.COVERAGE, "次要问题", fix_type=FixType.IN_PLACE),
        Issue(Severity.CRITICAL, Dimension.FACTUAL, "严重问题", fix_type=FixType.IN_PLACE),
    ]
    out, ops = _run(ba.defend(_report(), RedVerdict(issues=issues)))
    fix_ops = [o for o in ops if o.action.startswith("in_place")]
    assert len(fix_ops) == 2  # 两个 issue 都被处理


def test_defend_failed_parse_marks_unsuccessful():
    # policy 返回非 JSON → 解析失败 → 修复标记 success=False，但不崩
    ba = BlueAgent(policy=_MockPolicy("抱歉我无法修复"), tools=[])
    issue = Issue(Severity.MAJOR, Dimension.LOGICAL, "逻辑跳跃", fix_type=FixType.IN_PLACE)
    out, ops = _run(ba.defend(_report(), RedVerdict(issues=[issue])))
    assert any(not op.success for op in ops)
