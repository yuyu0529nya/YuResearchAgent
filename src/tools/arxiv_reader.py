"""
论文阅读工具 (ArxivReaderTool) — 支持自动级联 / Crossref / ArXiv / Semantic Scholar / OpenAlex

设计理由：
  ArXiv API 在中国大陆访问不稳定（经常超时/断开）。
  Semantic Scholar API 国内可达且免费，申请 Key 后 rate limit 更高。
  OpenAlex API 国内可达、完全免费、无需 Key，覆盖 2 亿+ 论文。
  通过 .env 中的 ARXIV_READER_BACKEND 切换后端，零源码修改。

后端对比：
  - crossref:           DOI 元数据稳定，免费无需 Key，但不保证提供摘要
  - arxiv:              论文库最全，但国内需 VPN
  - semantic_scholar:   国内可达，覆盖 2 亿+ 论文，含引用数据
  - openalex:           国内可达，完全免费无需 Key，元数据丰富
"""
from __future__ import annotations

import asyncio
import re
import json
import random
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urlencode

import aiohttp
from bs4 import BeautifulSoup

from ..utils.env_config import get_env

__all__ = ["ArxivReaderTool"]


_GENERIC_ACADEMIC_TERMS = {
    "academic", "algorithm", "analysis", "architecture", "benchmark", "citation",
    "comparison", "evaluation", "framework", "large", "language", "llm", "model",
    "paper", "publication", "report", "research", "study", "survey", "system",
    "technical", "technology", "the", "and", "for", "with", "using",
}


def _academic_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_.+-]*|\d+(?:\.\d+)?", (text or "").lower()))


class ArxivReaderTool:
    """论文读取工具：支持 ArXiv / Semantic Scholar / OpenAlex 三后端。

    配置优先从 .env / .env.local 读取：
      - ARXIV_READER_BACKEND: "auto" | "openalex" | "crossref" | "arxiv" | "semantic_scholar"（默认 auto）
      - ARXIV_API_ENDPOINT:    ArXiv API 端点（一般不需要改）
      - SEMANTIC_SCHOLAR_API_KEY: Semantic Scholar API Key（免费申请，可选）
      - OPENALEX_EMAIL:        OpenAlex 可选邮箱（提高 rate limit，建议填写）
    """

    name: str = "arxiv_reader"
    description: str = (
        "Read paper metadata from academic databases. "
        "Supports Crossref, ArXiv, Semantic Scholar, and OpenAlex backends. "
        "Input: {'paper_id': str(optional), 'query': str(optional), 'max_results': int(default=3)}. "
        "Output: list of paper metadata dicts."
    )

    def __init__(self, backend: str | None = None, use_mock: bool = False, delay_ms: tuple[int, int] = (50, 200)) -> None:
        self.backend = (backend or get_env("ARXIV_READER_BACKEND", "auto")).lower().strip()
        self.use_mock = use_mock
        self.delay_ms = delay_ms

        # ArXiv 配置
        self.arxiv_base_url = get_env("ARXIV_API_ENDPOINT", "http://export.arxiv.org/api/query")

        # Semantic Scholar 配置
        self.ss_api_key = get_env("SEMANTIC_SCHOLAR_API_KEY")
        self.ss_base_url = "https://api.semanticscholar.org/graph/v1"

        # OpenAlex 配置
        self.openalex_email = get_env("OPENALEX_EMAIL", "")
        self.openalex_base_url = "https://api.openalex.org"

        # Crossref 配置（免费无需 Key，部分网络环境比 ArXiv/OpenAlex 稳定）
        self.crossref_email = get_env("CROSSREF_EMAIL", "")
        self.crossref_base_url = "https://api.crossref.org"

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "description": "Provide either paper_id for lookup or query for search.",
                    "properties": {
                        "paper_id": {
                            "type": "string",
                            "description": "ArXiv paper ID or Semantic Scholar paper ID, e.g. '1706.03762'",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 3,
                        },
                    },
                },
            },
        }

    async def execute(
        self, paper_id: str | None = None, query: str | None = None, max_results: int = 3
    ) -> dict[str, Any]:
        if self.use_mock:
            return await self._mock_execute(paper_id, query, max_results)

        if self.backend == "auto":
            return await self._auto_execute(paper_id, query, max_results)
        if self.backend == "crossref":
            return await self._crossref_execute(paper_id, query, max_results)
        if self.backend == "semantic_scholar":
            return await self._semantic_scholar_execute(paper_id, query, max_results)
        if self.backend == "openalex":
            return await self._openalex_execute(paper_id, query, max_results)
        return await self._arxiv_execute(paper_id, query, max_results)

    async def _auto_execute(
        self,
        paper_id: str | None,
        query: str | None,
        max_results: int,
    ) -> dict[str, Any]:
        """Use OpenAlex for scholarly relevance, with bounded retry and DOI fallback."""
        # OpenAlex direct work IDs accept OpenAlex IDs and DOI URLs, not arXiv IDs.
        openalex_id = paper_id
        openalex_query = query
        if paper_id and not (paper_id.upper().startswith("W") or paper_id.startswith("10.")):
            openalex_id = None
            openalex_query = paper_id

        fetch_limit = max_results if paper_id else min(10, max(max_results * 3, max_results))
        attempts: list[dict[str, Any]] = []
        raw_result = await self._openalex_curl_execute(openalex_id, openalex_query, fetch_limit)
        result = self._filter_search_result(raw_result, openalex_query, max_results)
        attempts.append(result)
        if result.get("papers"):
            result["source"] = "auto:openalex"
            result["transport"] = "curl"
            return result

        # A second transport is useful for a network failure, but not when the
        # first transport already returned the same irrelevant ranking.
        if not raw_result.get("papers"):
            raw_result = await self._openalex_execute(openalex_id, openalex_query, fetch_limit)
            result = self._filter_search_result(raw_result, openalex_query, max_results)
            attempts.append(result)
            if result.get("papers"):
                result["source"] = "auto:openalex"
                result["transport"] = "aiohttp"
                return result

        crossref_id = paper_id if paper_id and paper_id.startswith("10.") else None
        raw_crossref = await self._crossref_execute(
            crossref_id,
            query or (paper_id if crossref_id is None else None),
            fetch_limit,
        )
        crossref = self._filter_search_result(
            raw_crossref,
            query or (paper_id if crossref_id is None else None),
            max_results,
        )
        if crossref.get("papers"):
            crossref["source"] = "auto:crossref"
            crossref["warning"] = "OpenAlex returned no papers; Crossref fallback used"
            return crossref
        return {
            "source": "auto:openalex+crossref",
            "query": query or paper_id,
            "papers": [],
            "error": "; ".join(
                str(result.get("error"))
                for result in [*attempts, crossref]
                if result.get("error")
            )
            or "Academic search returned no relevant papers",
        }

    @classmethod
    def _filter_search_result(
        cls,
        result: dict[str, Any],
        query: str | None,
        max_results: int,
    ) -> dict[str, Any]:
        """Filter metadata-only fallbacks that match generic academic terms only.

        OpenAlex and Crossref occasionally rank an unrelated item because both
        the query and title contain words such as "technical report" or
        "benchmark". Named model/version tokens are therefore treated as the
        discriminating part of the query. Direct ID lookups pass ``query=None``
        and remain untouched.
        """
        payload = dict(result)
        papers = [paper for paper in result.get("papers", []) if isinstance(paper, dict)]
        if not query:
            payload["papers"] = papers[:max_results]
            return payload

        query_tokens = {
            token
            for token in _academic_tokens(query)
            if token not in _GENERIC_ACADEMIC_TERMS
            and not (token.isdigit() and 1900 <= int(token) <= 2100)
        }
        if not query_tokens:
            payload["papers"] = papers[:max_results]
            return payload

        ranked: list[tuple[float, int, dict[str, Any]]] = []
        for index, paper in enumerate(papers):
            title_tokens = _academic_tokens(str(paper.get("title", "")))
            content_tokens = title_tokens | _academic_tokens(str(paper.get("summary", ""))[:2000])
            overlap = query_tokens & content_tokens
            if not overlap:
                continue
            ratio = len(overlap) / len(query_tokens)
            has_identifier = any(
                any(char.isdigit() for char in token) or any(char in token for char in ".+-")
                for token in overlap
            )
            if len(overlap) < 2 and ratio < 0.25 and not has_identifier:
                continue
            title_ratio = len(query_tokens & title_tokens) / len(query_tokens)
            ranked.append((title_ratio * 0.7 + ratio * 0.3, -index, paper))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        payload["papers"] = [paper for _, _, paper in ranked[:max_results]]
        filtered_count = len(papers) - len(payload["papers"])
        if filtered_count:
            payload["filtered_irrelevant"] = filtered_count
        return payload

    async def _openalex_curl_execute(
        self,
        paper_id: str | None,
        query: str | None,
        max_results: int,
    ) -> dict[str, Any]:
        """OpenAlex transport fallback for networks that reset aiohttp TLS."""
        if paper_id:
            url = f"{self.openalex_base_url}/works/{paper_id}"
        else:
            url = f"{self.openalex_base_url}/works?{urlencode({'search': query or '', 'per-page': max_results})}"
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
            return {"source": "openalex", "query": query or paper_id, "papers": []}
        except (asyncio.TimeoutError, OSError) as exc:
            return {
                "source": "openalex",
                "query": query or paper_id,
                "papers": [],
                "error": f"OpenAlex curl error: {type(exc).__name__}: {exc}",
            }

        body, separator, status_bytes = stdout.rpartition(marker)
        status = status_bytes.decode("ascii", errors="ignore").strip() if separator else ""
        if process.returncode != 0 or status != "200":
            return {
                "source": "openalex",
                "query": query or paper_id,
                "papers": [],
                "error": "OpenAlex curl HTTP "
                f"{status or 'unknown'}: {stderr.decode('utf-8', errors='ignore')[:200]}",
            }
        try:
            data = json.loads(body.decode("utf-8", errors="ignore"))
        except json.JSONDecodeError as exc:
            return {
                "source": "openalex",
                "query": query or paper_id,
                "papers": [],
                "error": f"OpenAlex returned invalid JSON: {exc}",
            }
        items = [data] if paper_id else data.get("results", [])
        return {
            "source": "openalex",
            "query": query or paper_id,
            "papers": [
                self._openalex_paper_to_dict(item)
                for item in items[:max_results]
                if isinstance(item, dict)
            ],
        }

    async def _crossref_execute(
        self, paper_id: str | None, query: str | None, max_results: int
    ) -> dict[str, Any]:
        """Search Crossref for DOI-backed scholarly metadata."""
        headers = {
            "User-Agent": (
                f"YuResearchAgent/0.1 (mailto:{self.crossref_email})"
                if self.crossref_email
                else "YuResearchAgent/0.1"
            )
        }
        if paper_id:
            url = f"{self.crossref_base_url}/works/{paper_id}"
            params = None
        else:
            url = f"{self.crossref_base_url}/works"
            params = {
                "query.bibliographic": query or "",
                "rows": max(1, min(max_results, 10)),
            }
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as response:
                    data = await response.json(content_type=None)
                    if response.status != 200:
                        return {
                            "source": "crossref",
                            "query": query or paper_id,
                            "papers": [],
                            "error": f"Crossref API error: HTTP {response.status}",
                        }
        except Exception as exc:
            return {
                "source": "crossref",
                "query": query or paper_id,
                "papers": [],
                "error": f"Crossref network error: {type(exc).__name__}: {exc}",
            }

        message = data.get("message", {})
        items = [message] if paper_id and isinstance(message, dict) else message.get("items", [])
        papers = [self._crossref_paper_to_dict(item) for item in items if isinstance(item, dict)]
        return {
            "source": "crossref",
            "query": query or paper_id,
            "papers": papers[:max_results],
        }

    @staticmethod
    def _crossref_paper_to_dict(data: dict[str, Any]) -> dict[str, Any]:
        title_value = data.get("title", "")
        title = title_value[0] if isinstance(title_value, list) and title_value else str(title_value or "")
        authors = []
        for author in data.get("author", [])[:10]:
            name = " ".join(
                part for part in (str(author.get("given", "")), str(author.get("family", ""))) if part
            )
            if name:
                authors.append(name)
        date_parts = (
            data.get("published", {}).get("date-parts")
            or data.get("published-online", {}).get("date-parts")
            or data.get("created", {}).get("date-parts")
            or []
        )
        published = "-".join(str(part) for part in date_parts[0]) if date_parts else ""
        abstract_html = str(data.get("abstract", ""))
        abstract = BeautifulSoup(abstract_html, "html.parser").get_text(" ", strip=True)
        abstract = re.sub(r"\s+", " ", abstract)
        doi = str(data.get("DOI", ""))
        return {
            "id": doi,
            "title": title,
            "authors": authors,
            "summary": abstract,
            "published": published,
            "pdf_url": str(data.get("URL") or (f"https://doi.org/{doi}" if doi else "")),
            "source": "crossref",
            "citation_count": data.get("is-referenced-by-count"),
            "publisher": data.get("publisher", ""),
            "container_title": (data.get("container-title") or [""])[0],
        }

    # ------------------------------------------------------------------
    # Mock 模式
    # ------------------------------------------------------------------
    async def _mock_execute(
        self, paper_id: str | None, query: str | None, max_results: int
    ) -> dict[str, Any]:
        await asyncio.sleep(random.randint(*self.delay_ms) / 1000.0)

        mock_papers = [
            {
                "id": "1706.03762",
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
                "summary": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
                "published": "2017-06-12",
                "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
                "source": "arxiv_mock",
            },
            {
                "id": "1810.04805",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
                "summary": "We introduce a new language representation model called BERT...",
                "published": "2018-10-11",
                "pdf_url": "https://arxiv.org/pdf/1810.04805.pdf",
                "source": "arxiv_mock",
            },
            {
                "id": "2303.18223",
                "title": "Large Language Models: A Survey",
                "authors": ["Wayne Xin Zhao", "Kun Zhou", "Junyi Li"],
                "summary": "This survey reviews the recent advances in large language models...",
                "published": "2023-03-31",
                "pdf_url": "https://arxiv.org/pdf/2303.18223.pdf",
                "source": "arxiv_mock",
            },
        ]

        if paper_id:
            papers = [p for p in mock_papers if p["id"] == paper_id]
        else:
            q = (query or "").lower()
            papers = [p for p in mock_papers if q in p["title"].lower() or q in p["summary"].lower()]

        return {
            "source": "arxiv_mock",
            "query": query or paper_id,
            "papers": papers[:max_results],
        }

    # ------------------------------------------------------------------
    # ArXiv 后端（论文最全，国内需 VPN）
    # ------------------------------------------------------------------
    async def _arxiv_execute(
        self, paper_id: str | None, query: str | None, max_results: int
    ) -> dict[str, Any]:
        if paper_id:
            search_query = f"id:{paper_id}"
        else:
            search_query = f"all:{query}"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.arxiv_base_url, params=params, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    text = await resp.text()
        except Exception as e:
            return {
                "source": "arxiv_api",
                "query": query or paper_id,
                "papers": [],
                "error": f"ArXiv API 网络错误（国内访问可能需 VPN）。建议切换到 semantic_scholar 后端："
                        f"在 .env 中设置 ARXIV_READER_BACKEND=semantic_scholar。原始错误: {e}",
            }

        # 解析 Atom XML
        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            preview = text[:200].replace("\n", " ")
            return {
                "source": "arxiv_api",
                "query": query or paper_id,
                "papers": [],
                "error": f"ArXiv API 返回了无法解析的内容。可能是服务暂时不可用或网络问题。"
                        f"内容预览: {preview}... (原始错误: {e})",
            }

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            paper = {
                "id": (entry.find("atom:id", ns).text or "").split("/")[-1],
                "title": (entry.find("atom:title", ns).text or "").strip().replace("\n", " "),
                "summary": (entry.find("atom:summary", ns).text or "").strip(),
                "published": entry.find("atom:published", ns).text or "",
                "pdf_url": "",
                "source": "arxiv_api",
            }
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    paper["pdf_url"] = link.get("href", "")
                    break
            authors = []
            for author in entry.findall("atom:author", ns):
                name_el = author.find("atom:name", ns)
                if name_el is not None:
                    authors.append(name_el.text or "")
            paper["authors"] = authors
            papers.append(paper)

        return {
            "source": "arxiv_api",
            "query": query or paper_id,
            "papers": papers,
        }

    # ------------------------------------------------------------------
    # Semantic Scholar 后端（国内可达，免费）
    # 申请 Key: https://www.semanticscholar.org/product/api#api-key-form
    # ------------------------------------------------------------------
    async def _semantic_scholar_execute(
        self, paper_id: str | None, query: str | None, max_results: int
    ) -> dict[str, Any]:
        headers = {}
        if self.ss_api_key:
            headers["x-api-key"] = self.ss_api_key

        try:
            if paper_id:
                # 直接按 ID 查询
                url = f"{self.ss_base_url}/paper/{paper_id}"
                params = {"fields": "title,authors,year,abstract,url,citationCount"}
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            return {
                                "source": "semantic_scholar",
                                "query": paper_id,
                                "papers": [],
                                "error": f"Semantic Scholar API 错误: {data.get('message', resp.status)}",
                            }
                        paper = self._ss_paper_to_dict(data)
                        return {
                            "source": "semantic_scholar",
                            "query": paper_id,
                            "papers": [paper],
                        }
            else:
                # 搜索查询
                url = f"{self.ss_base_url}/paper/search"
                params = {
                    "query": query,
                    "fields": "title,authors,year,abstract,url,citationCount",
                    "limit": max_results,
                }
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            return {
                                "source": "semantic_scholar",
                                "query": query,
                                "papers": [],
                                "error": f"Semantic Scholar API 错误: {data.get('message', resp.status)}",
                            }
                        papers = [self._ss_paper_to_dict(p) for p in data.get("data", [])]
                        return {
                            "source": "semantic_scholar",
                            "query": query,
                            "papers": papers,
                        }
        except Exception as e:
            return {
                "source": "semantic_scholar",
                "query": query or paper_id,
                "papers": [],
                "error": f"Semantic Scholar 网络错误: {e}",
            }

    @staticmethod
    def _ss_paper_to_dict(data: dict) -> dict:
        """将 Semantic Scholar 原始数据转为统一格式。"""
        authors = []
        for a in data.get("authors", [])[:10]:
            name = a.get("name", "")
            if name:
                authors.append(name)

        return {
            "id": data.get("paperId", "")[:20],
            "title": data.get("title", ""),
            "authors": authors,
            "summary": data.get("abstract", "") or "",
            "published": str(data.get("year", "")),
            "pdf_url": data.get("url", ""),
            "source": "semantic_scholar",
            "citation_count": data.get("citationCount"),
        }

    # ------------------------------------------------------------------
    # OpenAlex 后端（国内可达，完全免费，无需 Key）
    # 文档: https://docs.openalex.org/
    # ------------------------------------------------------------------
    async def _openalex_execute(
        self, paper_id: str | None, query: str | None, max_results: int
    ) -> dict[str, Any]:
        headers = {
            "User-Agent": "yu-research-agent",
            "Accept-Encoding": "gzip, deflate",  # 避免 brotli 解码问题
        }
        if self.openalex_email:
            headers["mailto"] = self.openalex_email

        try:
            if paper_id:
                # 直接按 ID 查询（支持 OpenAlex ID 或 DOI）
                url = f"{self.openalex_base_url}/works/{paper_id}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            return {
                                "source": "openalex",
                                "query": paper_id,
                                "papers": [],
                                "error": f"OpenAlex API 错误: {data.get('message', resp.status)}",
                            }
                        paper = self._openalex_paper_to_dict(data)
                        return {
                            "source": "openalex",
                            "query": paper_id,
                            "papers": [paper],
                        }
            else:
                # 搜索查询
                url = f"{self.openalex_base_url}/works"
                params = {
                    "search": query,
                    "per-page": max_results,
                }
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            return {
                                "source": "openalex",
                                "query": query,
                                "papers": [],
                                "error": f"OpenAlex API 错误: {data.get('message', resp.status)}",
                            }
                        papers = [self._openalex_paper_to_dict(r) for r in data.get("results", [])]
                        return {
                            "source": "openalex",
                            "query": query,
                            "papers": papers,
                        }
        except Exception as e:
            return {
                "source": "openalex",
                "query": query or paper_id,
                "papers": [],
                "error": f"OpenAlex 网络错误: {e}",
            }

    @staticmethod
    def _openalex_paper_to_dict(data: dict) -> dict:
        """将 OpenAlex 原始数据转为统一格式。"""
        authors = []
        for a in data.get("authorships", [])[:10]:
            author_info = a.get("author", {})
            name = author_info.get("display_name", "")
            if name:
                authors.append(name)

        # OpenAlex 的 abstract 是倒排索引，简单处理为空或从 summary 取
        summary = ""
        ab = data.get("abstract_inverted_index")
        if ab:
            # 倒排索引还原为近似文本（按词频排序不够精确，这里简单拼接）
            words = []
            for word, positions in ab.items():
                for pos in positions:
                    while len(words) <= pos:
                        words.append("")
                    words[pos] = word
            summary = " ".join(words)

        # PDF 链接
        pdf_url = ""
        oa = data.get("open_access", {})
        if oa:
            pdf_url = oa.get("oa_url", "") or oa.get("pdf_url", "")
        if not pdf_url:
            # 尝试从 best_oa_location 取
            loc = data.get("best_oa_location", {})
            if loc:
                pdf_url = loc.get("pdf_url", "") or loc.get("landing_page_url", "")

        return {
            "id": (data.get("id") or "").split("/")[-1],
            "title": data.get("display_name", ""),
            "authors": authors,
            "summary": summary,
            "published": str(data.get("publication_year", "")),
            "pdf_url": pdf_url,
            "source": "openalex",
            "citation_count": data.get("cited_by_count"),
        }
