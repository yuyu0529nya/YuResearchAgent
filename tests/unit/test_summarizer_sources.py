"""
tests/unit/test_summarizer_sources.py
SummarizerAgent._collect_sources 单测：从子结果轨迹提取去重来源，
arxiv 论文保留作者/年份(引用质量优化的核心)。无 LLM/网络。
"""
import asyncio
import time

from src.agents.summarizer import SummarizerAgent
from src.orchestrator.schemas import AgentResult, AgentStatus, SubTask, TaskType


class _P:
    tools = None

    def __call__(self, messages):
        return {"content": ""}


def _agent():
    return SummarizerAgent("sum", _P(), [])


def _result(traj, status=AgentStatus.SUCCESS, tid="t1"):
    return AgentResult(task_id=tid, status=status, output="x", confidence=0.8, trajectory=traj)


def test_collect_web_sources():
    r = _result([{"role": "tool", "result": {"results": [
        {"url": "https://a.com", "title": "A", "snippet": "s"},
        {"url": "https://b.com", "title": "B"},
    ]}}])
    srcs = _agent()._collect_sources([r])
    assert len(srcs) == 2
    assert {s["url"] for s in srcs} == {"https://a.com/", "https://b.com/"}


def test_collect_paper_keeps_authors_and_year():
    r = _result([{"role": "tool", "result": {"papers": [
        {"title": "Attention Is All You Need",
         "authors": ["Vaswani", "Shazeer", "Parmar", "Uszkoreit"],
         "published": "2017-06-12", "pdf_url": "https://arxiv.org/pdf/1706.03762"},
    ]}}])
    s = _agent()._collect_sources([r])[0]
    assert s["title"] == "Attention Is All You Need"
    assert s["year"] == "2017"
    assert "Vaswani" in s["authors"] and "et al." in s["authors"]  # >3 作者截断
    assert s["url"] == "https://arxiv.org/abs/1706.03762"


def test_collect_dedup_by_url():
    traj = [{"role": "tool", "result": {"results": [
        {"url": "https://x.com", "title": "X"},
        {"url": "https://x.com", "title": "X dup"},
    ]}}]
    assert len(_agent()._collect_sources([_result(traj)])) == 1


def test_collect_deduplicates_arxiv_url_variants() -> None:
    trajectory = [{"role": "tool", "result": {"results": [
        {"url": "https://arxiv.org/pdf/1706.03762v7.pdf", "title": "Attention PDF"},
        {"url": "https://www.arxiv.org/html/1706.03762", "title": "Attention HTML"},
    ]}}]

    sources = _agent()._collect_sources([_result(trajectory)])

    assert len(sources) == 1
    assert sources[0]["url"] == "https://arxiv.org/abs/1706.03762"


def test_collect_skips_failed_results():
    r = _result([{"role": "tool", "result": {"results": [{"url": "https://a.com", "title": "A"}]}}],
                status=AgentStatus.FAILED)
    assert _agent()._collect_sources([r]) == []


def test_summarizer_policy_call_does_not_block_event_loop():
    class _SlowPolicy:
        tools = None

        def __call__(self, messages):
            time.sleep(0.2)
            return {"content": "A sufficiently long report statement. Overall Confidence: 0.8"}

    async def _run() -> None:
        agent = SummarizerAgent("sum", _SlowPolicy(), [])
        task = SubTask("sum", TaskType.ANALYZE, "synthesize")
        result = _result([], tid="r1")
        await asyncio.wait_for(
            agent.run(task, {"query": "q", "results": [result]}),
            timeout=0.05,
        )

    try:
        asyncio.run(_run())
    except asyncio.TimeoutError:
        pass
    else:
        raise AssertionError("threaded policy call should remain cancellable by wait_for")
