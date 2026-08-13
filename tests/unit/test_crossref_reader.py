from __future__ import annotations

from src.tools.arxiv_reader import ArxivReaderTool


def test_explicit_backend_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("ARXIV_READER_BACKEND", "crossref")
    assert ArxivReaderTool(backend="openalex").backend == "openalex"


def test_academic_filter_rejects_generic_title_collision() -> None:
    result = {
        "source": "crossref",
        "papers": [
            {
                "title": "Benchmark Imagery FY11 Technical Report",
                "summary": "A remote-sensing benchmark report.",
            },
            {
                "title": "Qwen2.5 Technical Report",
                "summary": "Architecture and benchmark results for Qwen2.5.",
            },
        ],
    }

    filtered = ArxivReaderTool._filter_search_result(
        result,
        "Qwen2.5 architecture benchmark technical report",
        3,
    )

    assert [paper["title"] for paper in filtered["papers"]] == ["Qwen2.5 Technical Report"]
    assert filtered["filtered_irrelevant"] == 1


def test_direct_paper_lookup_is_not_relevance_filtered() -> None:
    result = {"papers": [{"title": "Any exact DOI result"}]}

    assert ArxivReaderTool._filter_search_result(result, None, 3)["papers"] == result["papers"]


def test_academic_filter_handles_chinese_topic_anchors() -> None:
    result = {
        "papers": [
            {
                "title": "STEM 课程设计与教师培养研究",
                "summary": "讨论跨学科课程和科学教师专业发展。",
            },
            {
                "title": "文莱与中国的外国投资关系",
                "summary": "分析油气产业和国有资本。",
            },
        ]
    }

    filtered = ArxivReaderTool._filter_search_result(
        result,
        "中美 STEM 教育课程设计与师资培养模式",
        3,
    )

    assert [paper["title"] for paper in filtered["papers"]] == [
        "STEM 课程设计与教师培养研究"
    ]
    assert filtered["papers"][0]["_relevance_score"] > 0.1


def test_academic_filter_requires_topic_anchor_in_title() -> None:
    result = {
        "papers": [
            {
                "title": "Personalized Education and Artificial Intelligence",
                "summary": "A broad review of machine learning in education around the world.",
            },
            {
                "title": "Tactile Sensing for Embodied Robot Learning",
                "summary": "A survey of touch sensing for dexterous robot manipulation.",
            },
        ]
    }

    filtered = ArxivReaderTool._filter_search_result(
        result,
        "world model tactile sensing embodied robot learning 2025",
        3,
    )

    assert [paper["title"] for paper in filtered["papers"]] == [
        "Tactile Sensing for Embodied Robot Learning"
    ]


def test_crossref_metadata_normalization() -> None:
    paper = ArxivReaderTool._crossref_paper_to_dict(
        {
            "DOI": "10.1000/test",
            "title": ["Claim-Evidence Graphs"],
            "author": [{"given": "Ada", "family": "Lovelace"}],
            "published": {"date-parts": [[2026, 7, 1]]},
            "URL": "https://doi.org/10.1000/test",
            "abstract": "<jats:p>A structured <b>evidence</b> abstract.</jats:p>",
            "is-referenced-by-count": 7,
            "publisher": "Example Press",
            "container-title": ["Research Journal"],
        }
    )

    assert paper["title"] == "Claim-Evidence Graphs"
    assert paper["authors"] == ["Ada Lovelace"]
    assert paper["published"] == "2026-7-1"
    assert paper["summary"] == "A structured evidence abstract."
    assert paper["citation_count"] == 7
