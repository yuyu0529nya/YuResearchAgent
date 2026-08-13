"""
合成 Agent (SummarizerAgent)

将多个 SubTask 的执行结果合成为结构化的研究报告。
区别于 ResearcherAgent 的多轮 tool-calling，Summarizer 是单轮长上下文生成任务：
  - 把所有子结果按置信度排序后拼接为上下文
  - 调用 LLM 一次性生成 Markdown 格式报告
  - 提取引用来源，计算整体置信度
"""
from __future__ import annotations

import asyncio
from collections import Counter
import json
import re
import time
from typing import Any

from .base_agent import BaseAgent
from ..evidence.store import (
    canonicalize_source_url,
    infer_source_year,
    source_quality,
    source_relevance,
)
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus, ResearchReport
from ..utils.tracing import trace_agent


__all__ = ["SummarizerAgent"]


class SummarizerAgent(BaseAgent):
    """合成 Agent：将子任务结果合成为最终研究报告。

    Attributes:
        max_output_tokens: 报告生成的最大 token 数（通过 policy.max_tokens 控制）。
    """

    def __init__(self, name: str, policy, tools: list | None = None) -> None:
        super().__init__(name, policy, tools)

    @trace_agent(name="summarizer.run", tags=["agent", "summarizer"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行合成任务。

        Args:
            task: 通常是一个特殊的 "synthesize" 类型任务。
            context: 全局上下文，必须包含 "results" 和 "query" 键。
                results: list[AgentResult]
                query: str 原始研究问题

        Returns:
            AgentResult，output 字段为 ResearchReport 实例。
        """
        query = context.get("query", "")
        as_of_date = str(context.get("as_of_date", "") or "").strip()
        results: list[AgentResult] = context.get("results", [])
        evidence_audit: dict = context.get("evidence_audit", {}) or {}
        evidence_sources: list[dict] = context.get("evidence_sources", []) or []
        evidence_sources = self._prioritize_evidence_sources(evidence_sources, evidence_audit)
        evidence_sources = self._select_synthesis_sources(query, evidence_sources, evidence_audit)
        cancellation_token = context.get("_cancellation_token")
        request_deadline = context.get("_request_deadline_monotonic")

        def _is_cancelled() -> bool:
            return bool(
                cancellation_token is not None
                and getattr(cancellation_token, "is_cancelled", False)
            )

        def _cancelled_result() -> AgentResult:
            reason = getattr(cancellation_token, "reason", "") or "Cancelled by user."
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.CANCELLED,
                output=reason,
                trajectory=[],
                token_usage=0,
                confidence=0.0,
            )

        if _is_cancelled():
            return _cancelled_result()

        if not results:
            report = ResearchReport(
                query=query,
                content="No sub-task results available to synthesize.",
                confidence=0.0,
            )
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=report,
                trajectory=[],
                token_usage=0,
                confidence=0.0,
            )

        # 构建 synthesis prompt
        prompt = self._build_synthesis_prompt(
            query,
            results,
            evidence_audit=evidence_audit,
            evidence_sources=evidence_sources,
            as_of_date=as_of_date,
        )
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]

        try:
            # 合成任务不需要工具调用，临时禁用 tools 避免模型进入 tool-calling 模式
            old_tools = getattr(self.policy, "tools", None)
            self.policy.tools = None
            try:
                call_with_timeout = getattr(self.policy, "call_with_timeout", None)
                if request_deadline is not None and callable(call_with_timeout):
                    remaining = max(
                        0.25,
                        float(request_deadline) - time.monotonic(),
                    )
                    response = await asyncio.to_thread(
                        call_with_timeout,
                        messages,
                        remaining,
                    )
                else:
                    response = await asyncio.to_thread(self.policy, messages)
            finally:
                self.policy.tools = old_tools
        except RuntimeError as e:
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=str(e),
                trajectory=[{"error": str(e)}],
                token_usage=0,
                confidence=0.0,
            )

        if _is_cancelled():
            return _cancelled_result()
        content = response.get("content", "") or ""
        if content.strip().lower().startswith("error:"):
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=content,
                trajectory=[{"error": content}],
                token_usage=0,
                confidence=0.0,
            )
        token_usage = len(content) // 3  # 简化估算

        # 解析报告内容，提取来源和置信度
        report = self._parse_report(query, content, results, evidence_sources=evidence_sources)

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.SUCCESS,
            output=report,
            trajectory=[{"role": "assistant", "content": content}],
            token_usage=token_usage,
            confidence=report.confidence,
        )

    def _system_prompt(self) -> str:
        return (
            "You are an expert research synthesizer. "
            "Your task is to integrate multiple research findings into a coherent, well-structured report. "
            "Use Markdown formatting. Cite sources explicitly. "
            "Respect the requested level of detail; normally write 1500-3000 Chinese characters "
            "or 1000-2000 English words. Include background, key findings, analysis, comparisons, and implications. "
            "DO NOT describe what you will do — directly output the synthesized report. "
            "At the end, provide an overall confidence score (0-1) and a summary of key sources."
        )

    def _collect_sources(
        self,
        results: list[AgentResult],
        evidence_sources: list[dict] | None = None,
    ) -> list[dict]:
        """从子结果轨迹提取去重来源：web(url/title) + arxiv 论文(含作者/年份)。"""
        sources: list[dict] = [dict(source) for source in (evidence_sources or [])]
        if evidence_sources is not None:
            return self._deduplicate_sources(sources)
        for r in results:
            if r.status != AgentStatus.SUCCESS:
                continue
            for step in r.trajectory:
                if step.get("role") != "tool" or not isinstance(step.get("result"), dict):
                    continue
                res = step["result"]
                if isinstance(res.get("results"), list):
                    for item in res["results"]:
                        if isinstance(item, dict) and item.get("url"):
                            quality, is_primary, host_domain = source_quality(item["url"], "web")
                            sources.append(
                                {
                                    "url": item["url"],
                                    "title": item.get("title", ""),
                                    "snippet": item.get("snippet", ""),
                                    "authors": "",
                                    "year": infer_source_year(
                                        item["url"],
                                        str(item.get("title", "")),
                                        str(item.get("snippet", "")),
                                    ),
                                    "task_id": r.task_id,
                                    "quality_score": quality,
                                    "is_primary": is_primary,
                                    "publisher": "",
                                    "metadata": {"host_domain": host_domain},
                                    "has_fulltext": False,
                                }
                            )
                if isinstance(res.get("papers"), list):
                    for p in res["papers"]:
                        if not isinstance(p, dict) or not (p.get("pdf_url") or p.get("title")):
                            continue
                        au = p.get("authors", [])
                        if isinstance(au, list):
                            au_str = ", ".join(au[:3]) + (" et al." if len(au) > 3 else "")
                        else:
                            au_str = str(au)
                        paper_url = p.get("pdf_url", p.get("url", ""))
                        quality, is_primary, host_domain = source_quality(paper_url, "paper")
                        sources.append(
                            {
                                "url": paper_url,
                                "title": p.get("title", ""),
                                "snippet": (p.get("summary", "") or "")[:200],
                                "authors": au_str,
                                "year": (p.get("published", "") or "")[:4]
                                or infer_source_year(paper_url, str(p.get("title", ""))),
                                "task_id": r.task_id,
                                    "quality_score": quality,
                                    "is_primary": is_primary,
                                    "publisher": str(p.get("publisher", "")),
                                    "metadata": {"host_domain": host_domain},
                                    "has_fulltext": False,
                            }
                        )
        return self._deduplicate_sources(sources)

    @staticmethod
    def _deduplicate_sources(sources: list[dict]) -> list[dict]:
        seen: dict[str, dict] = {}
        unique: list[dict] = []
        for s in sources:
            normalized = canonicalize_source_url(str(s.get("url", "")))
            if normalized:
                s["url"] = normalized
            key = normalized or str(s.get("title", "")).strip().lower()
            if not key:
                continue
            if key in seen:
                existing = seen[key]
                for field in ("title", "snippet", "authors", "year", "source_id", "publisher"):
                    if not existing.get(field) and s.get(field):
                        existing[field] = s[field]
                existing["quality_score"] = max(
                    float(existing.get("quality_score", 0.0) or 0.0),
                    float(s.get("quality_score", 0.0) or 0.0),
                )
                existing["is_primary"] = bool(existing.get("is_primary") or s.get("is_primary"))
                existing["has_fulltext"] = bool(existing.get("has_fulltext") or s.get("has_fulltext"))
                kinds = set(existing.get("evidence_kinds") or [])
                kinds.update(s.get("evidence_kinds") or [])
                if kinds:
                    existing["evidence_kinds"] = sorted(kinds)
                metadata = existing.get("metadata") if isinstance(existing.get("metadata"), dict) else {}
                incoming_metadata = s.get("metadata") if isinstance(s.get("metadata"), dict) else {}
                existing["metadata"] = {**incoming_metadata, **metadata}
                continue
            seen[key] = s
            unique.append(s)
        return unique

    @staticmethod
    def _select_synthesis_sources(
        query: str,
        evidence_sources: list[dict],
        evidence_audit: dict,
        limit: int = 20,
    ) -> list[dict]:
        """Expose only relevant, inspectable sources to the synthesis model."""
        supported_ids = {
            str(source_id)
            for claim in evidence_audit.get("claims", [])
            if claim.get("status") == "supported"
            for source_id in claim.get("source_ids", [])
        }
        candidates: list[dict] = []
        for source in evidence_sources:
            source_id = str(source.get("source_id", ""))
            title = str(source.get("title", ""))
            snippet = str(source.get("snippet", ""))
            url = str(source.get("url", ""))
            metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
            retrieval_relevance = float(metadata.get("retrieval_relevance", 0.0) or 0.0)
            direct_relevance, anchor_hits = source_relevance(query, f"{title} {snippet} {url}")
            title_relevance, title_anchor_hits = source_relevance(query, title)
            relevance = max(retrieval_relevance, direct_relevance)
            supported = source_id in supported_ids
            primary = bool(source.get("is_primary"))
            fulltext = bool(source.get("has_fulltext"))
            abstract = "abstract" in (source.get("evidence_kinds") or [])
            quality = float(source.get("quality_score", 0.5) or 0.5)
            inspectable = primary or fulltext or abstract or quality >= 0.65
            relevant = relevance >= 0.07 and (anchor_hits > 0 or retrieval_relevance >= 0.10)
            if not supported and (not inspectable or not relevant):
                continue
            if (
                not supported
                and not fulltext
                and title_anchor_hits == 0
                and max(retrieval_relevance, title_relevance) < 0.10
            ):
                continue
            if quality < 0.5 and not supported:
                continue
            host_domain = str(metadata.get("host_domain", ""))
            if not host_domain:
                _, _, host_domain = source_quality(url, str(source.get("source_type", "web")))
            candidate = dict(source)
            candidate["metadata"] = dict(metadata)
            candidate["metadata"].setdefault("host_domain", host_domain)
            candidates.append(candidate)

        # Quality-only global ranking starved broad landscape tasks whenever
        # narrower academic sub-tasks returned many papers. Allocate a small
        # round-robin quota to each original task before filling the remainder.
        # Gap-search tasks may fill unused slots but cannot displace a requested
        # dimension from the synthesis context.
        task_ids = sorted(
            {
                str(task_id)
                for source in candidates
                for task_id in (source.get("task_ids") or [])
                if task_id
                and not str(task_id).startswith("evidence_gap_")
                and str(task_id) != "final_report"
            },
            key=SummarizerAgent._task_sort_key,
        )
        selected: list[dict] = []
        selected_ids: set[str] = set()
        domain_counts: Counter[str] = Counter()
        title_keys: set[str] = set()

        def add_source(source: dict) -> bool:
            source_id = str(source.get("source_id", ""))
            identity = source_id or canonicalize_source_url(str(source.get("url", "")))
            if not identity or identity in selected_ids or len(selected) >= max(1, limit):
                return False
            metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
            host_domain = str(metadata.get("host_domain", ""))
            supported = source_id in supported_ids
            if domain_counts[host_domain] >= 3 and not supported:
                return False
            title_key = re.sub(r"\W+", "", str(source.get("title", ""))).lower()[:100]
            if title_key and title_key in title_keys and not supported:
                return False
            selected.append(dict(source))
            selected_ids.add(identity)
            domain_counts[host_domain] += 1
            if title_key:
                title_keys.add(title_key)
            return True

        for _ in range(3):
            for task_id in task_ids:
                for source in candidates:
                    if task_id in (source.get("task_ids") or []) and add_source(source):
                        break
        for source in candidates:
            if len(selected) >= max(1, limit):
                break
            add_source(source)
        return selected

    @staticmethod
    def _task_sort_key(task_id: str) -> tuple[str, int, str]:
        match = re.match(r"^(.*?)(\d+)$", task_id)
        if not match:
            return task_id, 0, task_id
        return match.group(1), int(match.group(2)), task_id

    @staticmethod
    def _prioritize_evidence_sources(
        evidence_sources: list[dict],
        evidence_audit: dict,
    ) -> list[dict]:
        """Rank sources by authority, full-text access, and verified claim support."""
        support_counts: Counter[str] = Counter()
        for claim in evidence_audit.get("claims", []):
            if claim.get("status") != "supported":
                continue
            support_counts.update(str(source_id) for source_id in claim.get("source_ids", []))

        def source_tier(source: dict) -> int:
            source_id = str(source.get("source_id", ""))
            supported = support_counts[source_id] > 0
            primary = bool(source.get("is_primary"))
            fulltext = bool(source.get("has_fulltext"))
            quality = float(source.get("quality_score", 0.5) or 0.5)
            if supported and primary:
                return 0
            if supported and fulltext:
                return 1
            if supported and quality >= 0.65:
                return 2
            if primary and fulltext:
                return 3
            if primary:
                return 4
            if fulltext and quality >= 0.75:
                return 5
            if fulltext:
                return 6
            if supported:
                return 7
            return 8

        indexed = list(enumerate(evidence_sources))
        indexed.sort(
            key=lambda item: (
                source_tier(item[1]),
                -support_counts[str(item[1].get("source_id", ""))],
                -int(bool(item[1].get("has_fulltext"))),
                -float(item[1].get("quality_score", 0.5) or 0.5),
                -int(item[1].get("evidence_count", 0) or 0),
                item[0],
            )
        )
        return [dict(source) for _, source in indexed]

    def _build_synthesis_prompt(
        self,
        query: str,
        results: list[AgentResult],
        evidence_audit: dict | None = None,
        evidence_sources: list[dict] | None = None,
        as_of_date: str = "",
    ) -> str:
        """构建合成 prompt，按置信度降序排列结果。"""
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

        parts = [
            f"# Research Question\n{query}\n",
            f"# Sub-task Results ({len(results)} total)\n",
        ]
        if as_of_date:
            parts.insert(
                1,
                (
                    f"# Temporal Context\nAs-of date: {as_of_date}. Include only information available "
                    "on or before this date; do not call eligible sources 'future' solely because they postdate "
                    "the model's pretraining knowledge.\n"
                ),
            )
        result_budget = 18_000
        per_result = max(1_500, result_budget // max(1, len(sorted_results)))
        for i, r in enumerate(sorted_results, 1):
            status_icon = "✓" if r.status == AgentStatus.SUCCESS else "✗"
            output = str(r.output or "")
            if len(output) > per_result:
                output = output[:per_result] + "\n[RESULT_TRUNCATED]"
            parts.append(
                f"## Result {i} [{status_icon}] (confidence: {r.confidence:.2f})\n"
                f"Task: {r.task_id}\n"
                f"Output:\n{output}\n"
            )

        sources = self._collect_sources(results, evidence_sources)
        if sources:
            parts.append("\n# 可用来源（引用时用这些编号 [N]，正文末尾参考文献也用它们）")
            for i, s in enumerate(sources[:25], 1):
                line = f"[{i}] {s['title'] or s['url']}"
                if s.get("authors"):
                    line += f" — {s['authors']}"
                elif s.get("publisher"):
                    line += f" — {s['publisher']}"
                if s.get("year"):
                    line += f"（{s['year']}）"
                if s.get("url") and s.get("title"):
                    line += f" — {s['url']}"
                labels = []
                if s.get("is_primary"):
                    labels.append("PRIMARY")
                if s.get("has_fulltext"):
                    labels.append("FULL_TEXT")
                elif "abstract" in (s.get("evidence_kinds") or []):
                    labels.append("ABSTRACT")
                else:
                    labels.append("SNIPPET_ONLY")
                if float(s.get("quality_score", 0.5) or 0.5) < 0.5:
                    labels.append("LOW_QUALITY")
                line += f" [{'|'.join(labels)}]"
                parts.append(line)
                excerpt = re.sub(
                    r"\s+",
                    " ",
                    str(s.get("evidence_excerpt") or s.get("snippet") or ""),
                ).strip()
                if excerpt:
                    parts.append(f"  Evidence ({s.get('evidence_kind') or 'snippet'}): {excerpt[:360]}")

        audit = evidence_audit or {}
        claims = audit.get("claims", []) if isinstance(audit, dict) else []
        audit_mode = str(audit.get("verification_mode", "heuristic"))
        audit_available = audit_mode not in {"hybrid_unavailable", "hybrid_partial"}
        if claims and not audit_available:
            parts.append(
                "\n# Claim-level Evidence Audit\n"
                f"Semantic audit status: {audit_mode}; reviewed "
                f"{audit.get('semantic_reviewed_count', 0)}/"
                f"{audit.get('semantic_candidate_count', 0)} eligible claim-evidence pairs. "
                "Do not treat unreviewed lexical NEI labels as proof that a claim is false or unsupported. State only "
                "claims directly entailed by the supplied metadata/excerpts, cite them precisely, and qualify "
                "anything else."
            )
        elif claims:
            source_numbers = {
                source.get("source_id"): index
                for index, source in enumerate(sources[:25], 1)
                if source.get("source_id")
            }
            parts.append(
                "\n# Claim-level Evidence Audit\n"
                f"Coverage: {audit.get('coverage', 0.0):.1%}; "
                f"Supported: {audit.get('supported_count', 0)}; "
                f"Refuted: {audit.get('refuted_count', 0)}; "
                f"Not enough evidence: {audit.get('not_enough_evidence_count', 0)}."
            )
            supported = [claim for claim in claims if claim.get("status") == "supported"]
            unresolved = [claim for claim in claims if claim.get("status") != "supported"]
            if supported:
                parts.append("\n## Verified claims (safe to state when cited)")
                for claim in supported[:25]:
                    refs = [
                        source_numbers[source_id]
                        for source_id in claim.get("source_ids", [])
                        if source_id in source_numbers
                    ]
                    ref_text = " ".join(f"[{number}]" for number in refs[:3])
                    excerpts = claim.get("evidence_excerpts", [])
                    excerpt = excerpts[0].get("text", "")[:260] if excerpts else ""
                    parts.append(f"- {claim.get('text', '')} {ref_text}\n  Evidence: {excerpt}")
            if unresolved:
                parts.append("\n## Claims requiring caution (do not present as established fact)")
                for claim in unresolved[:15]:
                    parts.append(
                        f"- [{claim.get('status', 'not_enough_evidence')}] {claim.get('text', '')} "
                        f"— {claim.get('reason', '')}"
                    )

        parts.append(
            "\n# Instructions\n"
            "1. Directly write the synthesized report based on the findings above. Do NOT say 'I will synthesize'.\n"
            "2. Respect the user's requested length. Otherwise target 2200-3500 Chinese characters or 1200-1800 English words.\n"
            "3. Mirror every explicit dimension in the question. For comparison questions, give one concise side-by-side "
            "table near the start, then analyze the causes and implications without repeating the table. Prefer direct, "
            "specific findings over generic background.\n"
            "4. Resolve any contradictions between sources.\n"
            "4b. When the semantic Claim-level Evidence Audit is available, treat it as a hard constraint: state "
            "SUPPORTED claims as facts only with their mapped source numbers; qualify or omit NOT_ENOUGH_EVIDENCE "
            "claims; explicitly report material REFUTED claims. If the audit explicitly says the semantic verifier "
            "was unavailable, validate directly against the supplied excerpts instead. Never upgrade a search "
            "snippet into stronger evidence than it provides.\n"
            "4c. Keep claims atomic: each sentence or semicolon-separated clause should contain one independently "
            "verifiable factual assertion. Do not describe tool failures, sub-task agreement, internal confidence, "
            "or what the evidence audit did; report the external findings and their limitations directly.\n"
            "4d. Source labels are hard quality guidance. Prefer PRIMARY and FULL_TEXT evidence. Do not rely on "
            "LOW_QUALITY, social-video, content-farm, or marketing sources when an authoritative source covers the "
            "topic; omit a claim when its only support is weak. SNIPPET_ONLY evidence supports only what the snippet "
            "states verbatim. Do not copy these internal labels into the bibliography.\n"
            "4e. Source relevance is mandatory: an authoritative publisher does not make an unrelated document valid. "
            "Use only sources whose title or supplied evidence directly matches the claim. Mention an evidence gap once, "
            "briefly, rather than repeating process limitations in multiple sections.\n"
            "4f. Source authors, organizations, years, and document ownership are facts too. State them only when they "
            "appear explicitly in the source catalog or evidence excerpt. A hosting domain is not necessarily the "
            "document's issuing organization, and a URL directory year is not necessarily the publication year.\n"
            "4g. A first-party company page can establish what that organization announced or released, but it is not "
            "independent validation of performance. Attribute vendor-reported metrics and distinguish them from "
            "peer-reviewed or independently reproduced results.\n"
            "5. 引用要**具体可验证**：关键论断后用上面「可用来源」里的**编号 [N]** 标注"
            "（如 [3]、[7]）；每个主要段落至少一处引用，**禁止用笼统的 [Result N]**。"
            "正文末尾必须有「## 参考来源」，把正文用到的每个 [N] 按"
            "「[N] 标题 — 作者/机构（年份） — 链接」完整、统一地列出（直接复用上面「可用来源」的条目）；"
            "不要列出正文未引用的来源，也不要猜测缺失的作者、机构或年份。\n"
            "6. End with: Overall Confidence: X.XX"
        )
        return "\n".join(parts)

    def _parse_report(
        self,
        query: str,
        content: str,
        results: list[AgentResult],
        evidence_sources: list[dict] | None = None,
    ) -> ResearchReport:
        """从 LLM 输出中解析 ResearchReport，并基于子任务成功率校准置信度。"""
        # 1. 从文本中提取 LLM 自评置信度
        llm_confidence = 0.5
        m = re.search(r"[Oo]verall\s+[Cc]onfidence[:\s]+(0\.\d+|1\.0|1)", content)
        if m:
            try:
                llm_confidence = float(m.group(1))
            except ValueError:
                pass

        # 2. 基于子任务成功率计算客观置信度
        total = len(results)
        success = sum(1 for r in results if r.status == AgentStatus.SUCCESS)
        success_rate = success / max(total, 1)

        # 3. 综合置信度 = LLM 自评 × 成功率开根（降低成功率的影响权重）
        confidence = llm_confidence * (success_rate ** 0.5)
        confidence = round(max(0.0, min(1.0, confidence)), 2)

        # 收集去重后的来源（含 arxiv 论文的作者/年份）
        unique_sources = self._collect_sources(results, evidence_sources)

        # 统计实际工具调用次数（遍历所有子任务的 trajectory）
        num_searches = sum(len([t for t in r.trajectory if t.get("role") == "tool"]) for r in results)

        return ResearchReport(
            query=query,
            content=content,
            sources=unique_sources,
            confidence=confidence,
            num_searches=num_searches,
        )
