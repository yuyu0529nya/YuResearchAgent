"""
网页搜索工具 — 支持 Yahoo/Brave/Wikipedia 自动级联及多种显式后端

设计理由：
  通过 .env 中的 SEARCH_BACKEND 切换后端，零源码修改。

后端对比：
  - auto:    Yahoo -> Brave -> Wikipedia，无 Key，并做相关性重排
  - serpapi: 每月 100 次免费，结果最全（Google 数据），国内可访问
  - bing:    微软搜索 API，国内稳定，需 Azure 订阅 Key
  - bocha:   博查AI搜索，国内索引最全，面向 AI Agent 优化
  - metaso:  秘塔AI搜索，中文语义强，有 research 多轮模式
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import base64
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

import aiohttp
from bs4 import BeautifulSoup

from ..utils.env_config import get_env

__all__ = ["WebSearchTool", "MockWebSearchTool", "BaseWebSearchTool"]


class BaseWebSearchTool(ABC):
    """网页搜索工具抽象基类。"""

    name: str = "web_search"
    description: str = (
        "Search the web for information. "
        "Supports automatic keyless Yahoo/Brave/Wikipedia search, SerpAPI, Bing API, 博查AI(bocha), and 秘塔AI(metaso) backends. "
        "Input: {'query': str, 'top_n': int(optional, default=5)}. "
        "Output: list of {'title': str, 'url': str, 'snippet': str}."
    )

    @abstractmethod
    async def execute(self, query: str, top_n: int = 5) -> dict[str, Any]:
        """执行搜索并返回结果。"""
        pass

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "top_n": {
                            "type": "integer",
                            "description": "返回结果数量",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }


class MockWebSearchTool(BaseWebSearchTool):
    """Mock 搜索工具：用于无网络环境的测试和演示。"""

    def __init__(self, delay_ms: tuple[int, int] = (50, 200)) -> None:
        self.delay_ms = delay_ms

    async def execute(self, query: str, top_n: int = 5) -> dict[str, Any]:
        await asyncio.sleep(random.randint(*self.delay_ms) / 1000.0)

        query_lower = query.lower()
        mock_db: dict[str, list[dict]] = {
            "transformer": [
                {
                    "title": "Attention Is All You Need",
                    "url": "https://arxiv.org/abs/1706.03762",
                    "snippet": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
                },
                {
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "url": "https://arxiv.org/abs/1810.04805",
                    "snippet": "BERT obtains new state-of-the-art results on eleven natural language processing tasks.",
                },
            ],
            "llm": [
                {
                    "title": "Large Language Models: A Survey",
                    "url": "https://arxiv.org/abs/2303.18223",
                    "snippet": "This survey reviews the recent advances in large language models, including pre-training, adaptation, and applications.",
                },
            ],
            "python": [
                {
                    "title": "Python Documentation",
                    "url": "https://docs.python.org/3/",
                    "snippet": "Official Python programming language documentation.",
                },
            ],
        }

        results: list[dict] = []
        for keyword, entries in mock_db.items():
            if keyword in query_lower:
                results.extend(entries)

        seen = set()
        unique = []
        for r in results:
            key = r["url"]
            if key not in seen:
                seen.add(key)
                unique.append(r)
        results = unique[:top_n]

        if not results:
            results = [
                {
                    "title": f"Mock result for '{query}'",
                    "url": "https://example.com/mock",
                    "snippet": "This is a mock search result for testing purposes.",
                }
            ]

        return {
            "query": query,
            "results": results,
            "total": len(results),
        }


class WebSearchTool(BaseWebSearchTool):
    """真实网页搜索工具：支持 keyless 与 API 搜索后端。

    配置优先从 .env / .env.local 读取：
      - SEARCH_BACKEND: auto | yahoo_html | brave_html | wikipedia | bing_html | duckduckgo | serpapi | bing | bocha | metaso
        （默认 auto）
      - SERPAPI_KEY / SERPAPI_ENDPOINT: SerpAPI 配置
      - BING_SEARCH_KEY / BING_SEARCH_ENDPOINT: Bing API 配置
    """

    _session: aiohttp.ClientSession | None = None
    _brave_lock = threading.Lock()
    _brave_last_request = 0.0

    def __init__(self, backend: str | None = None, api_key: str | None = None, api_endpoint: str | None = None) -> None:
        self.backend = (backend or get_env("SEARCH_BACKEND", "auto")).lower().strip()

        # SerpAPI 配置
        self.serpapi_key = api_key or get_env("SERPAPI_KEY")
        self.serpapi_endpoint = api_endpoint or get_env("SERPAPI_ENDPOINT", "https://serpapi.com/search")

        # Bing API 配置
        self.bing_key = api_key or get_env("BING_SEARCH_KEY")
        self.bing_endpoint = api_endpoint or get_env("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")

        # 博查AI 配置
        self.bocha_key = api_key or get_env("BOCHA_API_KEY")
        self.bocha_endpoint = api_endpoint or get_env("BOCHA_API_ENDPOINT", "https://api.bochaai.com/v1/web-search")

        # 秘塔AI 配置
        self.metaso_key = api_key or get_env("METASO_API_KEY")
        self.metaso_endpoint = api_endpoint or get_env("METASO_API_ENDPOINT", "https://metaso.cn/api/open/search/v2")

    def _get_session(self) -> aiohttp.ClientSession:
        """获取复用的 ClientSession，避免每次搜索新建连接。"""
        if WebSearchTool._session is None or WebSearchTool._session.closed:
            WebSearchTool._session = aiohttp.ClientSession(
                headers={"Accept-Encoding": "gzip, deflate"}
            )
        return WebSearchTool._session

    @classmethod
    async def close_session(cls) -> None:
        """关闭类级别的共享 session。应在程序退出前调用。"""
        if cls._session is not None and not cls._session.closed:
            await cls._session.close()
            cls._session = None

    def __del__(self):
        """析构时尝试关闭 session（同步环境回退）。"""
        if WebSearchTool._session is not None and not WebSearchTool._session.closed:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.close_session())
            except RuntimeError:
                # 无运行中的事件循环，忽略
                pass

    async def execute(self, query: str, top_n: int = 5) -> dict[str, Any]:
        if self.backend in ("auto", "keyless", "keyless_auto"):
            return await self._auto_execute(query, top_n)
        if self.backend in ("brave_html", "keyless_brave", "brave_web"):
            return await self._brave_html_execute(query, top_n)
        if self.backend in ("yahoo_html", "keyless_yahoo", "yahoo_web"):
            return await self._yahoo_html_execute(query, top_n)
        if self.backend in ("bing_html", "keyless_bing", "bing_web"):
            return await self._bing_html_execute(query, top_n)
        if self.backend in ("wikipedia", "wikipedia_api", "wiki"):
            return await self._wikipedia_execute(query, top_n)
        if self.backend == "bing":
            return await self._bing_execute(query, top_n)
        if self.backend == "bocha":
            return await self._bocha_execute(query, top_n)
        if self.backend == "metaso":
            return await self._metaso_execute(query, top_n)
        if self.backend in ("duckduckgo", "ddg", "ddgs"):
            return await self._duckduckgo_execute(query, top_n)
        return await self._serpapi_execute(query, top_n)

    async def _auto_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """Cascade across keyless engines and rerank their organic results."""
        yahoo = await self._yahoo_html_execute(query, max(top_n, 5))
        yahoo_results = self._rank_results(query, yahoo.get("results", []), top_n)
        if len(yahoo_results) >= min(2, top_n):
            yahoo["results"] = yahoo_results
            yahoo["total"] = len(yahoo_results)
            yahoo["source"] = "auto:yahoo_html"
            return yahoo

        brave = await self._brave_html_execute(query, top_n)
        combined = yahoo_results + brave.get("results", [])
        ranked = self._rank_results(query, combined, top_n)
        if len(ranked) >= min(2, top_n):
            return {
                "query": query,
                "results": ranked,
                "total": len(ranked),
                "source": "auto:yahoo_html+brave_html",
                **(
                    {"warning": str(yahoo.get("warning"))}
                    if yahoo.get("warning")
                    else {}
                ),
            }

        wikipedia = await self._wikipedia_execute(query, max(top_n, 5))
        combined += wikipedia.get("results", [])
        ranked = self._rank_results(query, combined, top_n)
        payload: dict[str, Any] = {
            "query": query,
            "results": ranked,
            "total": len(ranked),
            "source": "auto:yahoo_html+brave_html+wikipedia_api",
        }
        warnings = [
            warning
            for warning in (
                brave.get("warning") or brave.get("error"),
                yahoo.get("warning"),
                wikipedia.get("warning"),
            )
            if warning
        ]
        if warnings:
            payload["warning"] = "; ".join(warnings)
        return payload

    async def _wikipedia_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """Last-resort, explicitly labeled encyclopedia search fallback."""
        configured_language = get_env("WIKIPEDIA_LANGUAGE", "auto").lower().strip()
        if configured_language == "auto":
            language = "zh" if re.search(r"[\u4e00-\u9fff]", query or "") else "en"
        else:
            language = configured_language if re.fullmatch(r"[a-z-]{2,12}", configured_language) else "en"
        endpoint = f"https://{language}.wikipedia.org/w/api.php"
        url = f"{endpoint}?{urlencode({'action': 'query', 'list': 'search', 'srsearch': query, 'format': 'json', 'srlimit': max(1, min(top_n, 10))})}"
        marker = b"\n__YURA_HTTP_STATUS__:"
        try:
            process = await asyncio.create_subprocess_exec(
                "curl",
                "--location",
                "--compressed",
                "--silent",
                "--show-error",
                "--max-time",
                "12",
                "--user-agent",
                "YuResearchAgent/0.1",
                "--write-out",
                "\n__YURA_HTTP_STATUS__:%{http_code}",
                url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
        except FileNotFoundError:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": "Wikipedia API transport unavailable because curl is not installed",
            }
        except (asyncio.TimeoutError, OSError) as exc:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": f"Wikipedia API network error: {type(exc).__name__}: {exc}",
            }

        body, separator, status_bytes = stdout.rpartition(marker)
        status = status_bytes.decode("ascii", errors="ignore").strip() if separator else ""
        if process.returncode != 0 or status != "200":
            error = stderr.decode("utf-8", errors="ignore").strip()
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": f"Wikipedia API returned HTTP {status or 'unknown'}: {error[:200]}",
            }
        try:
            data = json.loads(body.decode("utf-8", errors="ignore"))
        except json.JSONDecodeError as exc:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": f"Wikipedia API returned invalid JSON: {exc}",
            }
        results = self._parse_wikipedia_response(data, language, top_n)
        payload: dict[str, Any] = {
            "query": query,
            "results": results,
            "total": len(results),
            "source": f"wikipedia_api:{language}",
        }
        if not results:
            payload["warning"] = "Wikipedia API returned no results"
        return payload

    @staticmethod
    def _parse_wikipedia_response(
        data: dict[str, Any],
        language: str,
        top_n: int,
    ) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        for item in data.get("query", {}).get("search", [])[:top_n]:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            title = str(item["title"])
            snippet = BeautifulSoup(str(item.get("snippet", "")), "html.parser").get_text(
                " ", strip=True
            )
            results.append(
                {
                    "title": title,
                    "url": f"https://{language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                    "snippet": snippet,
                }
            )
        return results

    async def _brave_html_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """Serialize keyless requests so parallel workers do not trigger 429s."""
        await asyncio.to_thread(self._brave_lock.acquire)
        try:
            delay = 1.5 - (time.monotonic() - self._brave_last_request)
            if delay > 0:
                await asyncio.sleep(delay)
            return await self._brave_html_execute_unlocked(query, top_n)
        finally:
            type(self)._brave_last_request = time.monotonic()
            self._brave_lock.release()

    async def _brave_html_execute_unlocked(self, query: str, top_n: int) -> dict[str, Any]:
        """Fetch Brave HTML through bounded curl when Python TLS is challenged.

        Brave currently returns a bot challenge to aiohttp/httpx in some regions,
        while the system curl TLS stack receives the public result page. The
        subprocess is argument-only (no shell), time-bounded, and output-capped.
        """
        endpoint = get_env("BRAVE_HTML_ENDPOINT", "https://search.brave.com/search")
        url = f"{endpoint}?{urlencode({'q': query, 'source': 'web'})}"
        marker = b"\n__YURA_HTTP_STATUS__:"
        user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        )
        try:
            process = await asyncio.create_subprocess_exec(
                "curl",
                "--location",
                "--compressed",
                "--silent",
                "--show-error",
                "--max-time",
                "12",
                "--user-agent",
                user_agent,
                "--header",
                "Accept-Language: en-US,en;q=0.9,zh-CN;q=0.8",
                "--write-out",
                "\n__YURA_HTTP_STATUS__:%{http_code}",
                url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
        except FileNotFoundError:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": "Brave HTML fallback unavailable because curl is not installed",
            }
        except (asyncio.TimeoutError, OSError) as exc:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": f"Brave HTML network error: {type(exc).__name__}: {exc}",
            }

        body, separator, status_bytes = stdout.rpartition(marker)
        status = status_bytes.decode("ascii", errors="ignore").strip() if separator else ""
        if process.returncode != 0 or status != "200":
            error = stderr.decode("utf-8", errors="ignore").strip()
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": f"Brave HTML returned HTTP {status or 'unknown'}: {error[:200]}",
            }
        if len(body) > 2_000_000:
            body = body[:2_000_000]
        results = self._parse_brave_html(body.decode("utf-8", errors="ignore"), top_n)
        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "brave_html",
        }

    @staticmethod
    def _parse_brave_html(html: str, top_n: int) -> list[dict[str, str]]:
        soup = BeautifulSoup(html or "", "html.parser")
        results: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in soup.select(".snippet"):
            title_node = item.select_one(".title")
            if title_node is None:
                continue
            link = title_node.find_parent("a") or item.select_one("a[href]")
            if link is None:
                continue
            url = str(link.get("href", ""))
            if not url.startswith(("http://", "https://")) or url in seen:
                continue
            snippet_node = item.select_one(".generic-snippet")
            seen.add(url)
            results.append(
                {
                    "title": title_node.get_text(" ", strip=True),
                    "url": url,
                    "snippet": snippet_node.get_text(" ", strip=True) if snippet_node else "",
                }
            )
            if len(results) >= top_n:
                break
        return results

    async def _yahoo_html_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """Keyless Yahoo search through a bounded, argument-only curl process."""
        endpoint = get_env("YAHOO_HTML_ENDPOINT", "https://search.yahoo.com/search")
        url = f"{endpoint}?{urlencode({'p': query, 'n': max(1, min(top_n, 10))})}"
        marker = b"\n__YURA_HTTP_STATUS__:"
        try:
            process = await asyncio.create_subprocess_exec(
                "curl",
                "--location",
                "--compressed",
                "--silent",
                "--show-error",
                "--max-time",
                "12",
                "--user-agent",
                (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
                ),
                "--header",
                "Accept-Language: en-US,en;q=0.9,zh-CN;q=0.8",
                "--write-out",
                "\n__YURA_HTTP_STATUS__:%{http_code}",
                url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
        except FileNotFoundError:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": "Yahoo HTML transport unavailable because curl is not installed",
            }
        except (asyncio.TimeoutError, OSError) as exc:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": f"Yahoo HTML network error: {type(exc).__name__}: {exc}",
            }

        body, separator, status_bytes = stdout.rpartition(marker)
        status = status_bytes.decode("ascii", errors="ignore").strip() if separator else ""
        if process.returncode != 0 or status != "200":
            error = stderr.decode("utf-8", errors="ignore").strip()
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": f"Yahoo HTML returned HTTP {status or 'unknown'}: {error[:200]}",
            }
        if len(body) > 2_000_000:
            body = body[:2_000_000]
        results = self._parse_yahoo_html(body.decode("utf-8", errors="ignore"), top_n)
        payload: dict[str, Any] = {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "yahoo_html",
        }
        if not results:
            payload["warning"] = "Yahoo HTML returned no organic results"
        return payload

    @classmethod
    def _parse_yahoo_html(cls, html: str, top_n: int) -> list[dict[str, str]]:
        soup = BeautifulSoup(html or "", "html.parser")
        results: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in soup.select(".algo"):
            link = item.select_one(".compTitle a[href]") or item.select_one("h3 a[href]")
            title_node = item.select_one("h3")
            if link is None or title_node is None:
                continue
            url = cls._unwrap_yahoo_url(str(link.get("href", "")))
            if not url.startswith(("http://", "https://")) or url in seen:
                continue
            snippet_node = item.select_one(".compText p") or item.select_one(".compText")
            seen.add(url)
            results.append(
                {
                    "title": title_node.get_text(" ", strip=True),
                    "url": url,
                    "snippet": snippet_node.get_text(" ", strip=True) if snippet_node else "",
                }
            )
            if len(results) >= top_n:
                break
        return results

    @staticmethod
    def _unwrap_yahoo_url(url: str) -> str:
        match = re.search(r"/RU=([^/]+)/RK=", url)
        if match:
            target = unquote(match.group(1))
            if target.startswith(("http://", "https://")):
                return target
        return url

    @classmethod
    def _rank_results(
        cls,
        query: str,
        results: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, str]]:
        """Deduplicate and suppress results with no lexical relation to the query."""
        query_tokens = cls._search_tokens(query)
        ranked: list[tuple[float, int, dict[str, str]]] = []
        seen: set[str] = set()
        for index, item in enumerate(results):
            url = str(item.get("url", ""))
            if not url or url in seen:
                continue
            seen.add(url)
            text = f"{item.get('title', '')} {item.get('snippet', '')} {url}"
            result_tokens = cls._search_tokens(text)
            overlap = len(query_tokens & result_tokens) / max(1, len(query_tokens))
            trusted = any(
                domain in urlsplit(url).netloc.lower()
                for domain in (
                    "arxiv.org", "openai.com", "anthropic.com", "deepmind.google",
                    "ai.google.dev", "github.com", "huggingface.co", ".gov", ".edu",
                )
            )
            if query_tokens and overlap < 0.08:
                continue
            score = overlap + (0.20 if trusted else 0.0)
            ranked.append(
                (
                    score,
                    -index,
                    {
                        "title": str(item.get("title", "")),
                        "url": url,
                        "snippet": str(item.get("snippet", "")),
                    },
                )
            )
        ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [item for _, _, item in ranked[:top_n]]

    @staticmethod
    def _search_tokens(text: str) -> set[str]:
        lowered = (text or "").lower()
        tokens = set(re.findall(r"[a-z][a-z0-9_.+-]+|\d+(?:\.\d+)?", lowered))
        for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
            tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
        return tokens

    async def _duckduckgo_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """DuckDuckGo 搜索（免费、无需 API Key）。依赖 ddgs 包：pip install ddgs"""
        def _search() -> list[dict]:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS  # 旧包名兼容
            timeout = float(get_env("DDGS_TIMEOUT_SECONDS", "6") or 6)
            engines = get_env("DDGS_ENGINES", "duckduckgo") or "duckduckgo"
            with DDGS(timeout=timeout) as ddgs:
                return list(ddgs.text(query, max_results=top_n, backend=engines))

        try:
            raw = await asyncio.to_thread(_search)
        except ImportError:
            return {"query": query, "results": [], "total": 0,
                    "error": "DuckDuckGo 后端需要 ddgs 包：pip install ddgs"}
        except Exception as e:
            return {"query": query, "results": [], "total": 0,
                    "warning": f"DuckDuckGo 搜索无结果: {type(e).__name__}: {e}"}

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", "") or r.get("url", ""),
                "snippet": r.get("body", "") or r.get("snippet", ""),
            }
            for r in raw[:top_n]
        ]
        return {"query": query, "results": results, "total": len(results)}

    async def _bing_html_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """Keyless Bing result-page search with a strict network timeout.

        This backend is intended as a zero-configuration fallback. It parses only
        the stable organic-result structure and never treats an empty page as a
        fatal tool error, allowing the agent to switch to academic search.
        """
        curl_result = await self._bing_html_curl_execute(query, top_n)
        if curl_result.get("results"):
            return curl_result

        endpoint = get_env("BING_HTML_ENDPOINT", "https://cn.bing.com/search")
        params = {"q": query, "count": max(1, min(top_n, 10))}
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        try:
            session = self._get_session()
            async with session.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                html = await response.text(errors="ignore")
                if response.status != 200:
                    return {
                        "query": query,
                        "results": [],
                        "total": 0,
                        "warning": f"Bing HTML returned HTTP {response.status}",
                    }
        except Exception as exc:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": f"Bing HTML network error: {type(exc).__name__}: {exc}",
            }

        results = self._parse_bing_html(html, top_n)
        payload: dict[str, Any] = {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "bing_html",
        }
        if not results:
            warnings = [
                curl_result.get("warning"),
                "Bing HTML returned no organic results",
            ]
            payload["warning"] = "; ".join(str(item) for item in warnings if item)
        return payload

    async def _bing_html_curl_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """Use the system TLS stack when Bing challenges Python HTTP clients."""
        endpoint = get_env("BING_HTML_CURL_ENDPOINT", "https://www.bing.com/search")
        url = f"{endpoint}?{urlencode({'q': query, 'count': max(1, min(top_n, 10))})}"
        marker = b"\n__YURA_HTTP_STATUS__:"
        user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        )
        try:
            process = await asyncio.create_subprocess_exec(
                "curl",
                "--location",
                "--compressed",
                "--silent",
                "--show-error",
                "--max-time",
                "12",
                "--user-agent",
                user_agent,
                "--header",
                "Accept-Language: en-US,en;q=0.9,zh-CN;q=0.8",
                "--write-out",
                "\n__YURA_HTTP_STATUS__:%{http_code}",
                url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
        except FileNotFoundError:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": "Bing HTML curl transport unavailable because curl is not installed",
            }
        except (asyncio.TimeoutError, OSError) as exc:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": f"Bing HTML curl error: {type(exc).__name__}: {exc}",
            }

        body, separator, status_bytes = stdout.rpartition(marker)
        status = status_bytes.decode("ascii", errors="ignore").strip() if separator else ""
        if process.returncode != 0 or status != "200":
            error = stderr.decode("utf-8", errors="ignore").strip()
            return {
                "query": query,
                "results": [],
                "total": 0,
                "warning": f"Bing HTML curl returned HTTP {status or 'unknown'}: {error[:200]}",
            }
        if len(body) > 2_000_000:
            body = body[:2_000_000]
        results = self._parse_bing_html(body.decode("utf-8", errors="ignore"), top_n)
        payload: dict[str, Any] = {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "bing_html:curl",
        }
        if not results:
            payload["warning"] = "Bing HTML curl returned no organic results"
        return payload

    @classmethod
    def _parse_bing_html(cls, html: str, top_n: int) -> list[dict[str, str]]:
        soup = BeautifulSoup(html or "", "html.parser")
        results: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in soup.select("li.b_algo"):
            link = item.select_one("h2 a")
            if link is None:
                continue
            url = cls._unwrap_bing_url(str(link.get("href", "")))
            title = link.get_text(" ", strip=True)
            caption = item.select_one(".b_caption p")
            snippet = caption.get_text(" ", strip=True) if caption else ""
            if not url.startswith(("http://", "https://")) or url in seen:
                continue
            seen.add(url)
            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= top_n:
                break
        return results

    @staticmethod
    def _unwrap_bing_url(url: str) -> str:
        """Decode Bing's ``u=a1<base64>`` redirect when present."""
        try:
            parsed = urlsplit(url)
            encoded = parse_qs(parsed.query).get("u", [""])[0]
            if encoded.startswith("a1"):
                payload = encoded[2:]
                payload += "=" * (-len(payload) % 4)
                decoded = base64.urlsafe_b64decode(payload).decode("utf-8")
                if decoded.startswith(("http://", "https://")):
                    return decoded
        except (ValueError, UnicodeDecodeError):
            pass
        return url

    async def _serpapi_execute(self, query: str, top_n: int) -> dict[str, Any]:
        if not self.serpapi_key:
            raise RuntimeError(
                "WebSearchTool (serpapi 后端) 需要 API Key。\n"
                "请在 .env 或 .env.local 中设置 SERPAPI_KEY，\n"
                "或构造函数传入: WebSearchTool(api_key='your_key')\n"
                "如需 Mock 模式，请显式使用 MockWebSearchTool()"
            )

        params = {
            "q": query,
            "num": top_n,
            "api_key": self.serpapi_key,
            "engine": "google",
            "gl": "us",
            "hl": "en",
        }

        try:
            session = self._get_session()
            async with session.get(
                self.serpapi_endpoint,
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        error_msg = data.get("error", f"HTTP {resp.status}")
                        return {
                            "query": query,
                            "results": [],
                            "total": 0,
                            "error": f"SerpAPI 错误: {error_msg}",
                        }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": f"SerpAPI 网络错误: {e}",
            }

        # 解析 SerpAPI 响应
        organic = data.get("organic_results", [])
        results = []
        for item in organic[:top_n]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "serpapi",
        }

    async def _bing_execute(self, query: str, top_n: int) -> dict[str, Any]:
        if not self.bing_key:
            raise RuntimeError(
                "WebSearchTool (bing 后端) 需要 API Key。\n"
                "请在 .env 或 .env.local 中设置 BING_SEARCH_KEY，\n"
                "或在 Azure Portal 创建 Bing Search v7 资源获取 Key。\n"
                "如需 Mock 模式，请显式使用 MockWebSearchTool()"
            )

        headers = {"Ocp-Apim-Subscription-Key": self.bing_key}
        params = {"q": query, "count": top_n, "mkt": "en-US"}

        try:
            session = self._get_session()
            async with session.get(
                self.bing_endpoint,
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        error_msg = data.get("message", f"HTTP {resp.status}")
                        return {
                            "query": query,
                            "results": [],
                            "total": 0,
                            "error": f"Bing API 错误: {error_msg}",
                        }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": f"Bing API 网络错误: {e}",
            }

        # 解析 Bing 响应
        web_pages = data.get("webPages", {}).get("value", [])
        results = []
        for item in web_pages[:top_n]:
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "bing",
        }

    async def _bocha_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """博查AI搜索后端。

        文档: https://open.bochaai.com
        特点: 国内网页索引最全，面向 AI Agent 和 RAG 优化，返回结构化摘要。
        """
        if not self.bocha_key:
            raise RuntimeError(
                "WebSearchTool (bocha 后端) 需要 API Key。\n"
                "请在 .env 或 .env.local 中设置 BOCHA_API_KEY，\n"
                "或访问 https://open.bochaai.com 注册获取。\n"
                "如需 Mock 模式，请显式使用 MockWebSearchTool()"
            )

        payload = {
            "query": query,
            "summary": True,
            "freshness": "noLimit",
            "count": top_n,
        }
        headers = {
            "Authorization": f"Bearer {self.bocha_key}",
            "Content-Type": "application/json",
        }

        try:
            session = self._get_session()
            async with session.post(
                self.bocha_endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    error_msg = data.get("message", f"HTTP {resp.status}")
                    return {
                        "query": query,
                        "results": [],
                        "total": 0,
                        "error": f"博查AI 错误: {error_msg}",
                    }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": f"博查AI 网络错误: {e}",
            }

        # 解析博查响应 — 兼容 web-search 和 ai-search 两种端点返回结构
        results: list[dict] = []

        # 结构 A: /v1/web-search → data.webPages.value[]
        web_pages = data.get("data", {}).get("webPages", {}).get("value", [])
        for item in web_pages[:top_n]:
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })

        # 结构 B: /v1/ai-search → data.messages[] content 里含引用
        if not results:
            messages = data.get("data", {}).get("messages", [])
            for msg in messages[:top_n]:
                content = msg.get("content", "")
                if content:
                    results.append({
                        "title": msg.get("role", "引用")[:30],
                        "url": "",
                        "snippet": content[:500],
                    })

        # 去重：同一篇文章的不同 URL（移动端/PC端/转发）会被当作多条结果
        results = self._deduplicate_results(results)

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "bocha",
        }

    def _deduplicate_results(self, results: list[dict]) -> list[dict]:
        """对搜索结果去重：基于规范化 URL 和清洗后的标题。"""
        from urllib.parse import urlparse
        import re

        seen_keys: set[str] = set()
        unique: list[dict] = []

        for r in results:
            raw_url = r.get("url", "")
            raw_title = r.get("title", "").strip()

            # --- URL 规范化 ---
            try:
                parsed = urlparse(raw_url)
                netloc = parsed.netloc.lower()
                path = parsed.path.lower().rstrip("/")

                # 去掉移动端前缀
                for prefix in ("m.", "wap.", "mobile.", "app."):
                    if netloc.startswith(prefix):
                        netloc = netloc[len(prefix):]
                        break
                # 去掉 www 前缀
                if netloc.startswith("www."):
                    netloc = netloc[4:]

                # 对常见新闻/博客站，只保留域名+路径（去掉查询参数）
                normalized_url = f"{netloc}{path}"
            except Exception:
                normalized_url = raw_url.lower().strip()

            # --- 标题清洗 ---
            # 去掉常见来源后缀，如 " - 虎嗅网"、"_CSDN博客"、"| 人人都是产品经理"
            cleaned_title = re.sub(
                r"[_\-\s|]*(CSDN博客|虎嗅网|人人都是产品经理|36氪|知乎|搜狐|新浪|网易|腾讯|今日头条|飞书云文档|简书|豆瓣|百度文库|原创力文档|道客巴巴|豆丁网|MBA智库文档|外唐智库|未来智库|中研网|中商产业研究院|三个皮匠报告|book118\.com|doc88\.com|docin\.com|mbalib\.com|askci\.com|chinairn\.com|vzkoo\.com|waitang\.com|sgpjbg\.com|toutiao\.com|sohu\.com|sina\.com|163\.com|qq\.com|ifeng\.com|huxiu\.com|36kr\.com|woshipm\.com|csdn\.net|zhihu\.com|juejin\.cn|segmentfault\.com|cnblogs\.com|简书|知乎专栏|百家号|大鱼号|企鹅号|新浪看点|一点资讯|趣头条|东方财富|雪球|同花顺|财联社|华尔街见闻|界面新闻|澎湃|新京报|南方周末|财新|第一财经|经济观察网|21世纪经济报道|新浪财经|腾讯财经|网易财经|凤凰财经|和讯网|中金在线|东方财富网|中国证券报|上海证券报|证券时报|证券日报|每日经济新闻|第一财经日报|经济参考报|人民日报|新华社|央视新闻|中央广播电视总台|中国日报|环球时报|参考消息|瞭望|半月谈|求是|学习强国|新华网|人民网|中国网|国际在线|中国新闻网|环球网等?)",
                "",
                raw_title,
                flags=re.IGNORECASE,
            ).strip()
            # 再去掉末尾的 " - "、" | "、"_"
            cleaned_title = re.sub(r"[_\-\s|]+$", "", cleaned_title).strip()

            # --- 去重键：优先用 URL，URL 为空时用清洗后的标题 ---
            key = normalized_url if normalized_url else cleaned_title.lower()
            if not key:
                unique.append(r)
                continue

            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique.append(r)

        return unique

    async def _metaso_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """秘塔AI搜索后端。

        文档: https://metaso.cn/open
        特点: 中文语义搜索强，支持 detail / concise / research 模式。
        """
        if not self.metaso_key:
            raise RuntimeError(
                "WebSearchTool (metaso 后端) 需要 API Key。\n"
                "请在 .env 或 .env.local 中设置 METASO_API_KEY，\n"
                "或访问 https://metaso.cn/open 注册获取。\n"
                "如需 Mock 模式，请显式使用 MockWebSearchTool()"
            )

        payload = {
            "question": query,
            "lang": "zh",
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.metaso_key}",
            "Content-Type": "application/json",
        }

        try:
            session = self._get_session()
            async with session.post(
                self.metaso_endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                if resp.status != 200 or data.get("errCode"):
                    error_msg = data.get("errMsg", f"HTTP {resp.status}")
                    return {
                        "query": query,
                        "results": [],
                        "total": 0,
                        "error": f"秘塔AI 错误: {error_msg}",
                    }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": f"秘塔AI 网络错误: {e}",
            }

        # 解析秘塔响应
        results: list[dict] = []
        result_data = data.get("data", {})

        # 1. 优先把 text 字段（秘塔 AI 整理的完整答案）作为高价值结果
        text = result_data.get("text", "")
        if text:
            results.append({
                "title": "秘塔AI搜索总结",
                "url": "",
                "snippet": text[:1500],  # 给足上下文，让 LLM 能直接总结
            })

        # 2. 附加参考文献列表（用于溯源）
        refs = result_data.get("references", [])
        for item in refs[:top_n]:
            snippet_parts = []
            if item.get("title"):
                snippet_parts.append(item["title"])
            if item.get("article_type"):
                snippet_parts.append(f"类型: {item['article_type']}")
            if item.get("date"):
                snippet_parts.append(f"日期: {item['date']}")
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": " | ".join(snippet_parts)[:500],
            })

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "metaso",
        }
