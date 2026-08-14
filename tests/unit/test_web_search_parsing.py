from __future__ import annotations

import asyncio
import base64

from src.tools.web_search import WebSearchTool


def test_parse_bing_html_extracts_organic_results() -> None:
    html = """
    <ol id="b_results">
      <li class="b_algo">
        <h2><a href="https://example.org/paper">Evidence Paper</a></h2>
        <div class="b_caption"><p>A primary-source abstract.</p></div>
      </li>
    </ol>
    """

    assert WebSearchTool._parse_bing_html(html, 5) == [
        {
            "title": "Evidence Paper",
            "url": "https://example.org/paper",
            "snippet": "A primary-source abstract.",
        }
    ]


def test_unwrap_bing_redirect_url() -> None:
    target = "https://arxiv.org/abs/2509.13312"
    encoded = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    redirect = f"https://www.bing.com/ck/a?u=a1{encoded}&ntb=1"

    assert WebSearchTool._unwrap_bing_url(redirect) == target


def test_parse_brave_html_extracts_results() -> None:
    html = """
    <div class="snippet">
      <a href="https://arxiv.org/abs/2412.15115">
        <div class="title">Qwen2.5 Technical Report</div>
      </a>
      <div class="generic-snippet">Introduces Qwen2.5 and its 18T-token corpus.</div>
    </div>
    """

    assert WebSearchTool._parse_brave_html(html, 5) == [
        {
            "title": "Qwen2.5 Technical Report",
            "url": "https://arxiv.org/abs/2412.15115",
            "snippet": "Introduces Qwen2.5 and its 18T-token corpus.",
        }
    ]


def test_result_ranking_removes_irrelevant_bing_noise() -> None:
    results = [
        {
            "title": "Qwen2.5 Technical Report",
            "url": "https://arxiv.org/abs/2412.15115",
            "snippet": "Qwen2.5 large language model technical report",
        },
        {
            "title": "How to type a star symbol",
            "url": "https://example.com/star",
            "snippet": "Word keyboard tutorial",
        },
    ]

    ranked = WebSearchTool._rank_results("Qwen2.5 Technical Report 2412.15115", results, 5)

    assert [item["url"] for item in ranked] == ["https://arxiv.org/abs/2412.15115"]


def test_result_ranking_prefers_primary_source_over_republisher() -> None:
    results = [
        {
            "title": "双减政策解读",
            "url": "https://www.sohu.com/a/123",
            "snippet": "双减政策对校外培训的要求",
        },
        {
            "title": "关于进一步减轻义务教育阶段学生负担的意见",
            "url": "https://www.gov.cn/zhengce/2021-07/24/content_5627132.htm",
            "snippet": "双减政策规范校外培训",
        },
    ]

    ranked = WebSearchTool._rank_results("双减 政策 校外培训", results, 5)

    assert ranked[0]["url"].startswith("https://www.gov.cn/")


def test_official_search_intent_rejects_generic_and_low_quality_results() -> None:
    results = [
        {
            "title": "中华人民共和国教育部政府门户网站",
            "url": "https://www.moe.gov.cn/",
            "snippet": "教育新闻与政务服务",
        },
        {
            "title": "双减政策解读视频",
            "url": "https://www.bilibili.com/video/BV1example",
            "snippet": "双减政策和校外培训解读",
        },
        {
            "title": "关于进一步减轻义务教育阶段学生作业负担和校外培训负担的意见",
            "url": "https://www.gov.cn/zhengce/2021-07/24/content_5627132.htm",
            "snippet": "双减政策官方原文，规范校外培训",
        },
    ]

    ranked = WebSearchTool._rank_results("双减 政策文件 官方原文", results, 5)

    assert [item["url"] for item in ranked] == [
        "https://www.gov.cn/zhengce/2021-07/24/content_5627132.htm"
    ]


def test_auto_search_stops_after_authoritative_ddgs_results(monkeypatch) -> None:
    tool = WebSearchTool("auto")
    calls: list[str] = []

    async def fake_ddgs(_query, _top_n, *, backend):
        calls.append(backend)
        return {
            "source": f"ddgs:{backend}",
            "results": [
                {
                    "title": "双减政策官方文件",
                    "url": "https://www.gov.cn/zhengce/example",
                    "snippet": "双减政策与校外培训",
                },
                {
                    "title": "教育部政策",
                    "url": "https://www.moe.gov.cn/srcsite/example",
                    "snippet": "科技教育与校外培训",
                },
                {
                    "title": "高校研究",
                    "url": "https://example.edu.cn/stem",
                    "snippet": "STEM 教育研究",
                },
            ],
        }

    async def unexpected_html(*_args, **_kwargs):
        raise AssertionError("authoritative DDGS results should avoid HTML fallbacks")

    monkeypatch.setattr(tool, "_ddgs_execute", fake_ddgs)
    monkeypatch.setattr(tool, "_yahoo_html_execute", unexpected_html)
    monkeypatch.setattr(tool, "_brave_html_execute", unexpected_html)
    monkeypatch.setattr(tool, "_bing_html_execute", unexpected_html)

    result = asyncio.run(tool.execute("双减 STEM 教育 校外培训", 5))

    assert calls == ["yandex"]
    assert result["source"] == "auto:ddgs:yandex"
    assert len(result["results"]) == 3


def test_auto_search_retries_official_domain_when_initial_results_are_weak(monkeypatch) -> None:
    tool = WebSearchTool("auto")
    calls: list[tuple[str, str]] = []

    async def fake_ddgs(query, _top_n, *, backend):
        calls.append((query, backend))
        if query.startswith("site:moe.gov.cn"):
            return {
                "source": "ddgs:yandex",
                "results": [
                    {
                        "title": "教育部印发义务教育课程方案和课程标准（2022年版）",
                        "url": "https://www.moe.gov.cn/jyb_xwfb/gzdt_gzdt/s5987/202204/example.html",
                        "snippet": "义务教育科学课程标准由教育部发布",
                    }
                ],
            }
        return {"source": f"ddgs:{backend}", "results": []}

    async def unexpected_html(*_args, **_kwargs):
        raise AssertionError("official-domain retry should avoid HTML fallbacks")

    monkeypatch.setattr(tool, "_ddgs_execute", fake_ddgs)
    monkeypatch.setattr(tool, "_yahoo_html_execute", unexpected_html)
    monkeypatch.setattr(tool, "_brave_html_execute", unexpected_html)
    monkeypatch.setattr(tool, "_bing_html_execute", unexpected_html)

    result = asyncio.run(tool.execute("义务教育科学课程标准 2022 官方原文", 5))

    assert calls[-1][0].startswith("site:moe.gov.cn")
    assert result["results"][0]["url"].startswith("https://www.moe.gov.cn/")


def test_ddgs_rejects_unknown_backend_without_silent_auto_fallback() -> None:
    tool = WebSearchTool("auto")

    result = asyncio.run(tool._ddgs_execute("query", 5, backend="removed-engine"))

    assert result["results"] == []
    assert result["error"] == "Unsupported DDGS text backend(s): removed-engine"


def test_ddgs_requests_are_serialized_across_parallel_workers(monkeypatch) -> None:
    tool = WebSearchTool("auto")
    active = 0
    peak_active = 0

    async def fake_unlocked(_query, _top_n, *, backend):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"query": "q", "results": [], "total": 0, "source": f"ddgs:{backend}"}

    monkeypatch.setattr(tool, "_ddgs_execute_unlocked", fake_unlocked)
    monkeypatch.setenv("DDGS_MIN_INTERVAL_SECONDS", "0")
    WebSearchTool._ddgs_last_request = 0.0

    async def run_parallel() -> None:
        await asyncio.gather(
            tool._ddgs_execute("first", 5, backend="yandex"),
            tool._ddgs_execute("second", 5, backend="duckduckgo"),
        )

    asyncio.run(run_parallel())

    assert peak_active == 1


def test_openrouter_citations_map_to_search_results() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "title": "Primary study",
                                "url": "https://example.edu/study",
                                "content": "A controlled evaluation of adaptive learning.",
                            },
                        },
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "title": "Duplicate",
                                "url": "https://example.edu/study",
                                "content": "ignored",
                            },
                        },
                    ]
                }
            }
        ]
    }

    assert WebSearchTool._openrouter_citation_results(payload, 3) == [
        {
            "title": "Primary study",
            "url": "https://example.edu/study",
            "snippet": "A controlled evaluation of adaptive learning.",
        }
    ]


def test_openrouter_search_defaults_to_a_tool_capable_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_SEARCH_MODEL", "")

    tool = WebSearchTool("openrouter")

    assert tool.openrouter_model == "openai/gpt-4.1-mini"


def test_parse_yahoo_html_extracts_and_unwraps_results() -> None:
    html = """
    <div class="dd algo algo-sr">
      <div class="compTitle">
        <a href="https://r.search.yahoo.com/x/RU=https%3A%2F%2Farxiv.org%2Fabs%2F2412.15115/RK=2/RS=x">
          <h3 class="title">Qwen2.5 Technical Report</h3>
        </a>
      </div>
      <div class="compText"><p>Architecture and benchmark details for Qwen2.5.</p></div>
    </div>
    """

    assert WebSearchTool._parse_yahoo_html(html, 5) == [
        {
            "title": "Qwen2.5 Technical Report",
            "url": "https://arxiv.org/abs/2412.15115",
            "snippet": "Architecture and benchmark details for Qwen2.5.",
        }
    ]


def test_parse_wikipedia_response_marks_traceable_article_urls() -> None:
    data = {
        "query": {
            "search": [
                {
                    "title": "Qwen",
                    "snippet": "A family of <span class='searchmatch'>language models</span>.",
                }
            ]
        }
    }

    assert WebSearchTool._parse_wikipedia_response(data, "en", 5) == [
        {
            "title": "Qwen",
            "url": "https://en.wikipedia.org/wiki/Qwen",
            "snippet": "A family of language models .",
        }
    ]
