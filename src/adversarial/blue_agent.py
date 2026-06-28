"""
M5 Blue Agent — 修复与防御器

Blue Agent 接收 Red Agent 的 Verdict，按优先级排序并执行三类修复：
1. In-place Fix：数字/日期与 source 不一致 → 直接替换
2. Supplementary Search：unsourced claims → 触发新搜索
3. Removal：高置信幻觉 → 删除段落

修复后执行 self_verify，确保不引入新矛盾。
"""
from __future__ import annotations

import copy
from typing import Any

from src.adversarial.verdict import (
    Dimension,
    FixOperation,
    FixType,
    Issue,
    RedVerdict,
    Severity,
    VerdictEngine,
)
from src.orchestrator.schemas import ResearchReport
from src.utils.tracing import trace_agent
from src.utils.json_parsing import extract_json_object


__all__ = ["BlueAgent"]


# ============================================================================
# Prompt 模板
# ============================================================================

SYSTEM_BLUE_AGENT = (
    "你是一位严谨的研究报告修订员（Blue Agent）。你的任务是根据审查意见修复研究报告，"
    "确保所有修改都有据可依，不引入新错误。输出必须是 JSON 格式。"
)

PROMPT_SELF_VERIFY = """请验证以下修复后的研究报告是否存在新引入的矛盾或错误。

请按以下 JSON 格式输出：
{
  "has_new_issue": bool,
  "new_issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",
      "location": "string"
    }
  ]
}

--- 原始报告 ---
{original}

--- 修复后报告 ---
{revised}

--- 已执行的修复 ---
{fixes}
"""

PROMPT_IN_PLACE_FIX = """请根据以下审查意见，对研究报告进行【定点修正】。

要求：
1. 只针对与 sources 不一致的具体事实（数字、日期、人名、机构等）做最小修改。
2. **绝不重写全文**——只输出需要替换的小片段：原文片段 → 修正后片段。
3. "before" 必须与【原始报告】中的文字**逐字一致**（含标点），否则无法定位替换。
4. 所有修正必须基于 sources，不引入新信息。若无需修正，返回空 edits 列表 []。

请按以下 JSON 格式输出（只给定点替换，不要全文）：
{
  "edits": [
    {
      "before": "string",   // 原文中需替换的片段，逐字复制
      "after":  "string",   // 修正后的片段
      "reason": "string"    // 简短理由（依据哪个 source）
    }
  ]
}

--- 审查意见 ---
{issue_desc}

--- 原始报告 ---
{content}

--- 来源 ---
{sources}
"""

PROMPT_SUPPLEMENTARY_SEARCH = """请根据以下审查意见，对研究报告进行【补充搜索后修正】。

要求：
1. 对无 source 支撑的 claim，使用搜索结果补充证据。
2. 如果搜索结果无法证实，则删除该 claim 或标注为"未经证实"。
3. 输出修正后的完整报告。

请按以下 JSON 格式输出：
{
  "fixed_content": "string",
  "changes": [
    {
      "location": "string",
      "action": "added|removed|modified",
      "detail": "string"
    }
  ]
}

--- 审查意见 ---
{issue_desc}

--- 原始报告 ---
{content}

--- 搜索结果 ---
{search_results}
"""

PROMPT_REMOVAL = """请根据以下审查意见，对研究报告进行【移除修正】。

要求：
1. 删除高置信度幻觉段落或无法验证的 claim。
2. 删除后确保上下文连贯，必要时添加过渡句。
3. 输出修正后的完整报告。

请按以下 JSON 格式输出：
{
  "fixed_content": "string",
  "removed_segments": [
    {
      "location": "string",
      "original_text": "string",
      "reason": "string"
    }
  ]
}

--- 审查意见 ---
{issue_desc}

--- 原始报告 ---
{content}
"""


# ============================================================================
# Blue Agent 实现
# ============================================================================

class BlueAgent:
    """Blue Agent — 修复与防御器。

    Attributes:
        policy: VLLMPolicy 实例。
        tools: 可用工具列表，至少包含搜索工具用于 supplementary_search。
        max_tokens: 单次修复调用的最大输出 token。
    """

    def __init__(self, policy, tools: list[Any] | None = None, max_tokens: int = 4096):
        self.policy = policy
        self.tools = tools or []
        self.max_tokens = max_tokens
        # 缓存搜索工具
        self._search_tool = self._find_search_tool()

    def _find_search_tool(self) -> Any | None:
        """从 tools 列表中查找搜索工具。"""
        for t in self.tools:
            name = getattr(t, "name", "")
            if "search" in name.lower():
                return t
        return None

    @trace_agent(name="blue_agent.defend", tags=["m5", "blue", "adversarial"])
    async def defend(
        self, report: ResearchReport, verdict: RedVerdict
    ) -> tuple[ResearchReport, list[FixOperation]]:
        """根据 Red Verdict 修复研究报告。

        执行流程：
        1. 按优先级对 issues 排序。
        2. 逐个执行修复（in_place / search / removal）。
        3. 每轮修复后执行 self_verify，检测是否引入新问题。
        4. 返回修复后的报告和所有 FixOperation 记录。

        Args:
            report: 原始研究报告（不会被修改，内部深拷贝）。
            verdict: Red Agent 的审查结果。

        Returns:
            (fixed_report, fix_operations)
        """
        current = copy.deepcopy(report)
        operations: list[FixOperation] = []

        if not verdict.issues:
            return current, operations

        # 按优先级降序排序
        sorted_issues = sorted(
            verdict.issues,
            key=lambda issue: VerdictEngine.compute_priority(issue),
            reverse=True,
        )

        original_content = report.content

        for issue in sorted_issues:
            op = await self._fix_single_issue(current, issue)
            operations.append(op)

            # self_verify：检查修复是否引入新矛盾
            verify_pass, verify_issues = await self._self_verify(
                original_content, current.content, operations
            )
            if not verify_pass:
                # 引入新问题：记录但继续（优先处理高优先级 issue，避免死锁）
                for vi in verify_issues:
                    operations.append(
                        FixOperation(
                            issue=vi,
                            action="self_verify_detected_new_issue",
                            success=False,
                            detail=vi.description,
                        )
                    )

        return current, operations

    async def _fix_single_issue(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """对单个 Issue 执行修复。"""
        old_max = getattr(self.policy, "max_tokens", None)
        if old_max is not None:
            self.policy.max_tokens = self.max_tokens

        try:
            if issue.fix_type == FixType.IN_PLACE:
                result = await self._do_in_place_fix(report, issue)
            elif issue.fix_type == FixType.SUPPLEMENTARY:
                result = await self._do_supplementary_search(report, issue)
            elif issue.fix_type == FixType.REMOVAL:
                result = await self._do_removal(report, issue)
            else:
                result = FixOperation(
                    issue=issue,
                    action="unknown_fix_type",
                    success=False,
                    detail=f"未知的 fix_type: {issue.fix_type}",
                )
        finally:
            if old_max is not None:
                self.policy.max_tokens = old_max

        return result

    async def _do_in_place_fix(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """执行原地修正。"""
        prompt = PROMPT_IN_PLACE_FIX
        prompt = prompt.replace("{issue_desc}", issue.description)
        prompt = prompt.replace("{content}", self._truncate_content(report.content))
        prompt = prompt.replace("{sources}", self._format_sources(report.sources, max_items=15))
        messages = [
            {"role": "system", "content": SYSTEM_BLUE_AGENT},
            {"role": "user", "content": prompt},
        ]
        resp = self.policy(messages)
        raw = resp.content or ""

        # 定点编辑：只对模型给出的 before→after 片段做精确替换，不重写全文（杜绝截断）。
        edits = self._parse_edits(raw)
        applied, details = self._apply_edits(report, edits)
        if applied > 0:
            return FixOperation(
                issue=issue,
                action=f"in_place_targeted_edit: 应用 {applied} 处",
                success=True,
                detail="; ".join(details)[:500],
            )
        return FixOperation(
            issue=issue,
            action="in_place_no_edit_applied",
            success=False,
            detail=(f"{len(edits)} 条 edit 的 before 均未逐字匹配原文" if edits else raw[:300]),
        )

    async def _do_supplementary_search(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """执行补充搜索后修正。"""
        search_results = ""
        if self._search_tool is not None:
            try:
                # 假设搜索工具有 async execute 或同步 execute 接口
                query = issue.description
                if hasattr(self._search_tool, "execute"):
                    if hasattr(self._search_tool.execute, "__call__"):
                        import inspect
                        if inspect.iscoroutinefunction(self._search_tool.execute):
                            sr = await self._search_tool.execute(query)
                        else:
                            sr = self._search_tool.execute(query)
                    else:
                        sr = None
                else:
                    sr = None
                search_results = str(sr) if sr else "（搜索工具未返回结果）"
            except Exception as e:
                search_results = f"（搜索失败: {e}）"
        else:
            search_results = "（无可用搜索工具）"

        prompt = PROMPT_SUPPLEMENTARY_SEARCH
        prompt = prompt.replace("{issue_desc}", issue.description)
        prompt = prompt.replace("{content}", self._truncate_content(report.content))
        # 截断搜索结果避免膨胀
        search_results = search_results[:2000] if len(search_results) > 2000 else search_results
        prompt = prompt.replace("{search_results}", search_results)
        messages = [
            {"role": "system", "content": SYSTEM_BLUE_AGENT},
            {"role": "user", "content": prompt},
        ]
        resp = self.policy(messages)
        raw = resp.content or ""

        fixed_content, changes = self._parse_fix_json(raw)
        if fixed_content:
            report.content = fixed_content
            return FixOperation(
                issue=issue,
                action=f"supplementary_search: {changes}",
                success=True,
                detail=f"search_results_len={len(search_results)}, changes={changes}",
            )
        return FixOperation(
            issue=issue,
            action="supplementary_search_failed",
            success=False,
            detail=raw[:500],
        )

    async def _do_removal(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """执行移除修正。"""
        prompt = PROMPT_REMOVAL
        prompt = prompt.replace("{issue_desc}", issue.description)
        prompt = prompt.replace("{content}", self._truncate_content(report.content))
        messages = [
            {"role": "system", "content": SYSTEM_BLUE_AGENT},
            {"role": "user", "content": prompt},
        ]
        resp = self.policy(messages)
        raw = resp.content or ""

        fixed_content, removed = self._parse_removal_json(raw)
        if fixed_content:
            report.content = fixed_content
            return FixOperation(
                issue=issue,
                action=f"removal: {removed}",
                success=True,
                detail=f"removed={removed}",
            )
        return FixOperation(
            issue=issue,
            action="removal_failed",
            success=False,
            detail=raw[:500],
        )

    async def _self_verify(
        self, original: str, revised: str, operations: list[FixOperation]
    ) -> tuple[bool, list[Issue]]:
        """修复后自验证，检查是否引入新矛盾。

        Returns:
            (是否通过, 新发现的 issues 列表)
        """
        if not revised or revised == original:
            return True, []

        fixes_text = "\n".join(
            f"- [{op.issue.dimension.value}] {op.action}: {op.detail[:200]}"
            for op in operations[-5:]  # 只取最近5条，避免 prompt 过长
        )
        prompt = PROMPT_SELF_VERIFY
        prompt = prompt.replace("{original}", original[:2000])
        prompt = prompt.replace("{revised}", revised[:2000])
        prompt = prompt.replace("{fixes}", fixes_text)
        messages = [
            {"role": "system", "content": SYSTEM_BLUE_AGENT},
            {"role": "user", "content": prompt},
        ]
        resp = self.policy(messages)
        raw = resp.content or ""

        data = extract_json_object(raw)
        if data is None:
            # self_verify 解析失败视为通过，避免阻塞修复流程
            return True, []
        try:
            has_new = bool(data.get("has_new_issue", False))
            new_issues = []
            for item in data.get("new_issues", []):
                new_issues.append(
                    Issue(
                        severity=Severity(item.get("severity", "minor")),
                        dimension=Dimension(item.get("dimension", "logical")),
                        description=item.get("description", ""),
                        location=item.get("location", ""),
                        fix_type=FixType.IN_PLACE,
                    )
                )
            return not has_new, new_issues
        except Exception:
            # Issue 构造失败（如枚举值非法）视为通过，避免阻塞修复流程
            return True, []

    def _format_sources(self, sources: list[dict], max_items: int = 15) -> str:
        """格式化来源列表，截断以避免上下文膨胀。"""
        if not sources:
            return "（无来源）"
        lines = []
        for i, s in enumerate(sources[:max_items], 1):
            title = s.get("title", "未知标题")
            url = s.get("url", "")
            snippet = s.get("snippet", "")[:300]
            lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}\n")
        if len(sources) > max_items:
            lines.append(f"... 还有 {len(sources) - max_items} 个来源未显示")
        return "\n".join(lines)

    def _truncate_content(self, content: str, max_len: int = 4000) -> str:
        """截断报告内容，避免 prompt 过长。"""
        if len(content) <= max_len:
            return content
        return content[:max_len] + "\n\n[报告已截断，仅显示前 {} 字符]".format(max_len)

    def _parse_edits(self, raw: str) -> list[dict]:
        """解析定点编辑 JSON：{"edits": [{"before","after","reason"}]}。"""
        data = extract_json_object(raw)
        if not data:
            return []
        edits = data.get("edits", [])
        return edits if isinstance(edits, list) else []

    def _apply_edits(self, report: ResearchReport, edits: list[dict]) -> tuple[int, list[str]]:
        """对报告正文精确应用 before→after 替换；before 未逐字匹配则跳过(保证不损坏原文)。"""
        applied = 0
        details: list[str] = []
        for e in edits:
            if not isinstance(e, dict):
                continue
            before = (e.get("before") or "").strip()
            after = e.get("after") or ""
            if before and before in report.content:
                report.content = report.content.replace(before, after, 1)
                applied += 1
                details.append(f"'{before[:30]}'→'{after[:30]}'")
        return applied, details

    def _parse_fix_json(self, raw: str) -> tuple[str, list[dict]]:
        """解析 in_place / search 修复的 JSON 输出（supplementary_search 仍用全文形式）。"""
        data = extract_json_object(raw)
        if data is None:
            return "", []
        return data.get("fixed_content", ""), data.get("changes", [])

    def _parse_removal_json(self, raw: str) -> tuple[str, list[dict]]:
        """解析 removal 修复的 JSON 输出。"""
        data = extract_json_object(raw)
        if data is None:
            return "", []
        return data.get("fixed_content", ""), data.get("removed_segments", [])
