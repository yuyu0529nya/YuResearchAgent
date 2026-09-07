from __future__ import annotations

import asyncio

import pytest

from src.tools.browser import BrowserTool


def test_arxiv_pdf_prefers_html_then_pdf_without_version_suffix() -> None:
    assert BrowserTool._candidate_urls("https://arxiv.org/pdf/2412.15115v2.pdf") == [
        ("https://arxiv.org/html/2412.15115", "fulltext"),
        ("https://arxiv.org/pdf/2412.15115.pdf", "fulltext"),
    ]


def test_arxiv_abs_marks_metadata_page_as_abstract_only() -> None:
    assert BrowserTool._candidate_urls("https://www.arxiv.org/abs/1706.03762") == [
        ("https://arxiv.org/html/1706.03762", "fulltext"),
        ("https://arxiv.org/abs/1706.03762", "abstract"),
    ]


def test_regular_page_has_single_fulltext_candidate() -> None:
    assert BrowserTool._candidate_urls("https://example.com/article") == [
        ("https://example.com/article", "fulltext")
    ]


def test_browser_rejects_private_and_credential_bearing_targets() -> None:
    with pytest.raises(ValueError, match="private or non-public"):
        asyncio.run(BrowserTool._validate_public_url("http://127.0.0.1:7860/"))
    with pytest.raises(ValueError, match="credentials"):
        asyncio.run(BrowserTool._validate_public_url("https://user:pass@example.com/"))
