from __future__ import annotations

import asyncio

import pytest

from src.tools.browser import BrowserTool


def test_arxiv_pdf_preserves_requested_version() -> None:
    assert BrowserTool._candidate_urls("https://arxiv.org/pdf/2412.15115v2.pdf") == [
        ("https://arxiv.org/html/2412.15115v2", "fulltext"),
        ("https://arxiv.org/pdf/2412.15115v2.pdf", "fulltext"),
        ("https://arxiv.org/abs/2412.15115v2", "abstract"),
    ]


def test_arxiv_abs_marks_metadata_page_as_abstract_only() -> None:
    assert BrowserTool._candidate_urls("https://www.arxiv.org/abs/1706.03762") == [
        ("https://arxiv.org/html/1706.03762", "fulltext"),
        ("https://arxiv.org/pdf/1706.03762.pdf", "fulltext"),
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


def test_redirect_validates_each_hop(monkeypatch) -> None:
    from src.tools.browser import _Redirect

    browser = BrowserTool()
    checked, fetched = [], []

    async def validate(url):
        checked.append(url)
        if "127.0.0.1" in url:
            raise ValueError("private target")

    async def fetch(url):
        fetched.append(url)
        raise _Redirect("http://127.0.0.1/admin")

    monkeypatch.setattr(browser, "_validate_public_url", validate)
    monkeypatch.setattr(browser, "_fetch_once", fetch)
    with pytest.raises(ValueError, match="private target"):
        asyncio.run(browser._fetch("https://example.com/paper"))
    assert checked == ["https://example.com/paper", "http://127.0.0.1/admin"]
    assert fetched == ["https://example.com/paper"]


def test_relative_redirect_reaches_document(monkeypatch) -> None:
    from src.tools.browser import _Redirect

    browser = BrowserTool()
    checked = []

    async def validate(url):
        checked.append(url)

    async def fetch(url):
        if url.endswith("/paper"):
            raise _Redirect("/download.pdf")
        return b"%PDF document", "application/pdf"

    monkeypatch.setattr(browser, "_validate_public_url", validate)
    monkeypatch.setattr(browser, "_fetch_once", fetch)
    assert asyncio.run(browser._fetch("https://example.com/paper"))[0] == b"%PDF document"
    assert checked == ["https://example.com/paper", "https://example.com/download.pdf"]


def test_redirect_loop_is_bounded(monkeypatch) -> None:
    from src.tools.browser import _Redirect

    browser = BrowserTool()
    calls = []

    async def validate(url):
        pass

    async def fetch(url):
        calls.append(url)
        raise _Redirect(url)

    monkeypatch.setattr(browser, "_validate_public_url", validate)
    monkeypatch.setattr(browser, "_fetch_once", fetch)
    with pytest.raises(ValueError, match="too many redirects"):
        asyncio.run(browser._fetch("https://example.com/paper"))
    assert len(calls) == 6


def test_curl_redirect_metadata_is_not_treated_as_body(monkeypatch) -> None:
    from src.tools.browser import _Redirect

    class Process:
        returncode = 0

        async def communicate(self):
            return b"Redirecting\n__YURA_BROWSER_META__:302\ttext/html\thttps://example.com/pdf", b""

    async def spawn(*args, **kwargs):
        assert "--location" not in args
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(_Redirect) as exc:
        asyncio.run(BrowserTool()._fetch_with_curl("https://example.com/paper"))
    assert exc.value.location == "https://example.com/pdf"


def test_cancelled_curl_is_killed_and_reaped(monkeypatch) -> None:
    class Process:
        returncode = None
        killed = False
        calls = 0

        async def communicate(self):
            self.calls += 1
            if self.calls == 1:
                raise asyncio.CancelledError()
            return b"", b""

        def kill(self):
            self.killed = True
            self.returncode = -9

    process = Process()

    async def spawn(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(BrowserTool()._fetch_with_curl("https://example.com/paper"))
    assert process.killed
    assert process.calls == 2
