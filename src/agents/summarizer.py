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
import json
import re
from typing import Any

from .base_agent import BaseAgent
from ..evidence.store import canonicalize_source_url
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
        results: list[AgentResult] = context.get("results", [])
        evidence_audit: dict = context.get("evidence_audit", {}) or {}
        evidence_sources: list[dict] = context.get("evidence_sources", []) or []
        evidence_sources = self._prioritize_evidence_sources(evidence_sources, evidence_audit)

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
                            sources.append(
                                {
                                    "url": item["url"],
                                    "title": item.get("title", ""),
                                    "snippet": item.get("snippet", ""),
                                    "authors": "",
                                    "year": "",
                                    "task_id": r.task_id,
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
                        sources.append(
                            {
                                "url": p.get("pdf_url", p.get("url", "")),
                                "title": p.get("title", ""),
                                "snippet": (p.get("summary", "") or "")[:200],
                                "authors": au_str,
                                "year": (p.get("published", "") or "")[:4],
                                "task_id": r.task_id,
                            }
                        )
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
                for field in ("title", "snippet", "authors", "year", "source_id"):
                    if not existing.get(field) and s.get(field):
                        existing[field] = s[field]
                continue
            seen[key] = s
            unique.append(s)
        return unique

    @staticmethod
    def _prioritize_evidence_sources(
        evidence_sources: list[dict],
        evidence_audit: dict,
    ) -> list[dict]:
        """Place sources used by supported claims first while preserving stable order."""
        preferred_ids: list[str] = []
        for claim in evidence_audit.get("claims", []):
            if claim.get("status") != "supported":
                continue
            preferred_ids.extend(claim.get("source_ids", []))
        rank = {source_id: index for index, source_id in enumerate(dict.fromkeys(preferred_ids))}
        indexed = list(enumerate(evidence_sources))
        indexed.sort(
            key=lambda item: (
                0 if item[1].get("source_id") in rank else 1,
                rank.get(item[1].get("source_id"), len(rank)),
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
    ) -> str:
        """构建合成 prompt，按置信度降序排列结果。"""
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

        parts = [
            f"# Research Question\n{query}\n",
            f"# Sub-task Results ({len(results)} total)\n",
        ]
        for i, r in enumerate(sorted_results, 1):
            status_icon = "✓" if r.status == AgentStatus.SUCCESS else "✗"
            parts.append(
                f"## Result {i} [{status_icon}] (confidence: {r.confidence:.2f})\n"
                f"Task: {r.task_id}\n"
                f"Output:\n{r.output}\n"
            )

        sources = self._collect_sources(results, evidence_sources)
        if sources:
            parts.append("\n# 可用来源（引用时用这些编号 [N]，正文末尾参考文献也用它们）")
            for i, s in enumerate(sources[:25], 1):
                line = f"[{i}] {s['title'] or s['url']}"
                if s.get("authors"):
                    line += f" — {s['authors']}"
                if s.get("year"):
                    line += f"（{s['year']}）"
                if s.get("url") and s.get("title"):
                    line += f" — {s['url']}"
                parts.append(line)

        audit = evidence_audit or {}
        claims = audit.get("claims", []) if isinstance(audit, dict) else []
        if claims:
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
            "2. Respect the user's requested length. Otherwise target 1500-3000 Chinese characters or 1000-2000 English words.\n"
            "3. Structure: Executive Summary → Background → Key Findings (with details) → Analysis → Comparisons → Implications → Conclusion.\n"
            "4. Resolve any contradictions between sources.\n"
            "4b. Treat the Claim-level Evidence Audit as a hard constraint: state SUPPORTED claims as facts only with "
            "their mapped source numbers; qualify or omit NOT_ENOUGH_EVIDENCE claims; explicitly report material "
            "REFUTED claims. Never upgrade a search snippet into stronger evidence than it provides.\n"
            "4c. Keep claims atomic: each sentence or semicolon-separated clause should contain one independently "
            "verifiable factual assertion. Do not describe tool failures, sub-task agreement, internal confidence, "
            "or what the evidence audit did; report the external findings and their limitations directly.\n"
            "5. 引用要**具体可验证**：关键论断后用上面「可用来源」里的**编号 [N]** 标注"
            "（如 [3]、[7]）；每个主要段落至少一处引用，**禁止用笼统的 [Result N]**。"
            "正文末尾必须有「## 参考来源」，把正文用到的每个 [N] 按"
            "「[N] 标题 — 作者/机构（年份） — 链接」完整、统一地列出（直接复用上面「可用来源」的条目）。\n"
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
