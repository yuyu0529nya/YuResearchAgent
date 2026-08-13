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
    assert all(s["publisher"] == "" for s in srcs)


def test_synthesis_prompt_includes_bounded_source_evidence_without_guessing_host_org():
    source = {
        "source_id": "src_1",
        "title": "Federal Strategic Plan",
        "url": "https://assets.example.gov/files/plan.pdf",
        "authors": "",
        "publisher": "",
        "year": "",
        "quality_score": 0.9,
        "is_primary": True,
        "has_fulltext": True,
        "evidence_kind": "full_text",
        "evidence_excerpt": "The Committee on STEM Education issues this five-year plan.",
    }

    prompt = _agent()._build_synthesis_prompt(
        "Federal STEM policy",
        [_result([], tid="r1")],
        evidence_sources=[source],
    )

    assert "Evidence (full_text): The Committee on STEM Education" in prompt
    assert "— assets.example.gov" not in prompt
    assert "A hosting domain is not necessarily" in prompt


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


def test_prioritize_sources_prefers_primary_over_weak_supported_snippet():
    sources = [
        {
            "source_id": "weak",
            "url": "https://douyin.com/video/1",
            "quality_score": 0.35,
            "is_primary": False,
            "has_fulltext": False,
        },
        {
            "source_id": "official",
            "url": "https://www.gov.cn/policy/1",
            "quality_score": 0.9,
            "is_primary": True,
            "has_fulltext": False,
        },
    ]
    audit = {"claims": [{"status": "supported", "source_ids": ["weak"]}]}

    ranked = _agent()._prioritize_evidence_sources(sources, audit)

    assert [source["source_id"] for source in ranked] == ["official", "weak"]


def test_select_synthesis_sources_blocks_irrelevant_primary_paper_and_weak_sites():
    query = "对比中美 STEM 教育政策、课程设计和师资培养，评估双减影响"
    sources = [
        {
            "source_id": "stem",
            "title": "Federal STEM Education Strategic Plan",
            "url": "https://www.nsf.gov/stem-plan.pdf",
            "snippet": "Federal STEM strategy and implementation",
            "publisher": "nsf.gov",
            "quality_score": 0.9,
            "is_primary": True,
            "has_fulltext": True,
        },
        {
            "source_id": "borneo",
            "title": "Foreign Investment and State Capitalism in Borneo",
            "url": "https://example.edu/paper.pdf",
            "snippet": "Oil, gas, investment, and national development",
            "publisher": "example.edu",
            "quality_score": 0.9,
            "is_primary": True,
            "has_fulltext": False,
            "evidence_kinds": ["abstract"],
        },
        {
            "source_id": "content-farm",
            "title": "STEM policy notes",
            "url": "https://doc88.com/p/1",
            "snippet": "STEM education policy",
            "publisher": "doc88.com",
            "quality_score": 0.35,
            "is_primary": False,
            "has_fulltext": True,
        },
    ]

    selected = _agent()._select_synthesis_sources(query, sources, {"claims": []})

    assert [source["source_id"] for source in selected] == ["stem"]


def test_select_synthesis_sources_preserves_each_original_task_before_gap_sources():
    query = "Embodied AI robotics world models tactile sensing task planning"
    sources = []
    for task_id in ("task_1", "task_2", "task_3", "task_4"):
        for index in range(1, 5):
            sources.append(
                {
                    "source_id": f"{task_id}_{index}",
                    "task_ids": [task_id],
                    "title": f"Embodied AI robotics evidence for {task_id} {index}",
                    "url": f"https://{task_id}.example.org/{index}",
                    "snippet": "world models tactile sensing task planning benchmark",
                    "quality_score": 0.9 if task_id != "task_1" else 0.65,
                    "is_primary": task_id != "task_1",
                    "has_fulltext": task_id != "task_1",
                    "metadata": {"retrieval_relevance": 0.9},
                }
            )
    sources.extend(
        {
            "source_id": f"gap_{index}",
            "task_ids": ["evidence_gap_r1_1"],
            "title": f"Embodied AI gap source {index}",
            "url": f"https://gap.example.org/{index}",
            "snippet": "world models tactile sensing task planning benchmark",
            "quality_score": 0.95,
            "is_primary": True,
            "has_fulltext": True,
            "metadata": {"retrieval_relevance": 0.95},
        }
        for index in range(1, 6)
    )

    ranked = _agent()._prioritize_evidence_sources(sources, {"claims": []})
    selected = _agent()._select_synthesis_sources(
        query,
        ranked,
        {"claims": []},
        limit=12,
    )

    selected_tasks = {
        task_id
        for source in selected
        for task_id in source.get("task_ids", [])
        if task_id.startswith("task_")
    }
    assert selected_tasks == {"task_1", "task_2", "task_3", "task_4"}
    assert sum(source["source_id"].startswith("task_1") for source in selected) == 3
    assert all(not source["source_id"].startswith("gap_") for source in selected)


def test_collect_with_empty_evidence_catalog_does_not_leak_trajectory_sources():
    trajectory = [{"role": "tool", "result": {"results": [
        {"url": "https://example.com/noise", "title": "Noise"},
    ]}}]

    assert _agent()._collect_sources([_result(trajectory)], evidence_sources=[]) == []


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
