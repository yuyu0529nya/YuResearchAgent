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
