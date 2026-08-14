"""Pure presentation helpers for structured run events."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .events import RunEvent


_STAGES = {
    "planning": ("规划", "拆解问题为子任务 DAG"),
    "dispatching": ("检索", "并发执行检索与核验任务"),
    "collecting": ("汇总", "归并结果并检查证据缺口"),
    "synthesizing": ("合成", "生成结构化研究报告"),
    "adversarial": ("对抗精修", "执行 Red-Blue 审查"),
    "evidence_refining": ("证据终审", "核验并门控报告修订"),
}

_TERMINAL_STATUSES = {
    "complete",
    "partial_timeout",
    "partial_failure",
    "cancelled",
    "cancelled_partial",
    "failed",
    "interrupted",
}


def new_run_view() -> dict[str, Any]:
    return {
        "current": "idle",
        "last_active": "idle",
        "status": "running",
        "tasks": {},
        "events": [],
        "evidence": {},
        "replan": 0,
    }


def apply_run_event(view: dict[str, Any], event: RunEvent | dict[str, Any]) -> dict[str, Any]:
    """Apply a typed event to a UI view model without parsing log text."""
    data = event.to_dict() if isinstance(event, RunEvent) else dict(event)
    kind = str(data.get("kind", ""))
    state = str(data.get("state", "") or "idle")
    payload = dict(data.get("payload") or {})
    message = str(data.get("message", ""))

    if state not in {"idle", "done", "failed"}:
        view["current"] = state
        view["last_active"] = state
    if kind == "run_started":
        view["status"] = "running"
    elif kind == "plan_created":
        tasks = view.setdefault("tasks", {})
        for task in payload.get("tasks", []):
            task_id = str(task.get("task_id", ""))
            if task_id:
                tasks[task_id] = {
                    "description": str(task.get("description", "")),
                    "type": str(task.get("type", "")),
                    "status": "waiting",
                }
    elif kind == "task_started":
        task_id = str(payload.get("task_id", ""))
        task = view.setdefault("tasks", {}).setdefault(task_id, {})
        task.update(
            {
                "status": "running",
                "type": str(payload.get("task_type", task.get("type", ""))),
                "description": message or task.get("description", ""),
            }
        )
    elif kind == "task_completed":
        task_id = str(payload.get("task_id", ""))
        task = view.setdefault("tasks", {}).setdefault(task_id, {})
        task["status"] = str(payload.get("status", "failed"))
        task["confidence"] = float(payload.get("confidence", 0.0) or 0.0)
        task["tool_calls"] = int(payload.get("tool_calls", 0) or 0)
    elif kind in {"evidence_snapshot", "final_audit_completed"}:
        view.setdefault("evidence", {}).update(payload)
    elif kind == "evidence_gap_started":
        view["evidence"]["gap_rounds"] = int(payload.get("round", 0) or 0)
    elif kind == "replan_started":
        view["replan"] = int(payload.get("round", 0) or 0)
    elif kind in {"revision_accepted", "revision_rejected"}:
        view["evidence"]["revision"] = {
            "attempted": True,
            "accepted": bool(payload.get("accepted")),
            "reason": message,
            "before": payload.get("before", {}),
            "after": payload.get("after", payload.get("before", {})),
        }
    elif kind == "revision_skipped":
        view["evidence"]["revision"] = {
            "attempted": False,
            "accepted": False,
            "reason": message,
        }
    elif kind in {"cancellation_requested", "cancellation_observed"}:
        view["status"] = "cancelling"
    elif kind == "run_failed":
        view["status"] = "failed"
    elif kind == "run_completed":
        view["status"] = str(payload.get("status") or "complete")

    label = _event_label(kind, message, payload)
    if label:
        events = view.setdefault("events", [])
        if not events or events[-1] != label:
            events.append(label)
            del events[:-12]
    return view


def _event_label(kind: str, message: str, payload: dict[str, Any]) -> str:
    if kind == "plan_created":
        return f"规划完成：{payload.get('task_count', 0)} 个子任务"
    if kind == "dispatch_layer_started":
        return f"执行第 {payload.get('layer', 0)}/{payload.get('layer_count', 0)} 层任务"
    if kind == "task_started":
        return f"{payload.get('task_id', 'task')} 开始执行"
    if kind == "task_completed":
        return f"{payload.get('task_id', 'task')} · {str(payload.get('status', '')).upper()}"
    if kind in {"evidence_snapshot", "final_audit_completed"}:
        return f"证据覆盖率 {float(payload.get('coverage', 0.0) or 0.0):.1%}"
    if kind == "evidence_gap_started":
        return f"启动第 {payload.get('round', 0)} 轮证据补检"
    if kind == "synthesis_started":
        return "开始合成研究报告"
    if kind == "context_compression_started":
        return "检查并压缩合成上下文"
    if kind == "context_compression_completed":
        before = int(payload.get("input_characters", 0) or 0)
        after = int(payload.get("output_characters", 0) or 0)
        fallback = " · 超时回退原文" if payload.get("status") == "timeout_fallback" else ""
        return f"合成上下文就绪 · {before:,} → {after:,} 字符{fallback}"
    if kind == "synthesis_completed":
        return f"报告合成 · {str(payload.get('status', '')).upper()}"
    if kind == "final_audit_started":
        return "开始最终 Claim 审计"
    if kind == "revision_started":
        return "开始证据约束修订"
    if kind == "revision_accepted":
        return "修订通过质量门控"
    if kind == "revision_rejected":
        return "修订未通过质量门控"
    if kind == "revision_skipped":
        return "未触发报告修订"
    if kind == "replan_started":
        return f"启动第 {payload.get('round', 0)} 次重规划"
    if kind == "planning_context_timeout":
        return "规划上下文超时，已无记忆降级继续"
    if kind == "cancellation_requested":
        return "收到停止请求"
    if kind == "cancellation_observed":
        return "已停止启动新的模型与工具调用"
    if kind == "run_completed":
        return f"运行结束 · {str(payload.get('status', '')).upper()}"
    if kind == "run_failed":
        return "运行失败"
    return message[:160] if kind == "runtime_error" else ""


def render_progress(
    view: dict[str, Any],
    *,
    elapsed: float,
    backend: str,
    adversarial: bool,
    run_id: str,
) -> str:
    pipeline = ["planning", "dispatching", "collecting", "synthesizing"]
    if adversarial:
        pipeline.append("adversarial")
    pipeline.append("evidence_refining")

    current = view.get("last_active") or view.get("current") or "idle"
    if current == "replanning":
        current = "dispatching"
    status = str(view.get("status") or "running")
    terminal = status in _TERMINAL_STATUSES
    successful_terminal = status == "complete"
    current_index = pipeline.index(current) if current in pipeline else -1

    status_label = {
        "running": "RUNNING",
        "cancelling": "STOPPING",
        "complete": "COMPLETE",
        "partial_timeout": "PARTIAL · TIMEOUT",
        "partial_failure": "PARTIAL · FAILURE",
        "cancelled": "CANCELLED",
        "cancelled_partial": "PARTIAL · CANCELLED",
        "failed": "FAILED",
        "interrupted": "INTERRUPTED",
    }.get(status, status.upper())
    lines = [
        f"`{status_label}` **{elapsed:.0f}s** · `{backend}`",
        f"`{run_id}`",
        "",
    ]

    for index, stage in enumerate(pipeline):
        name, description = _STAGES[stage]
        if successful_terminal or index < current_index:
            mark = "`OK`"
        elif index == current_index:
            mark = "`STOP`" if terminal else "`RUN`"
        else:
            mark = "`WAIT`"
        suffix = f" - {description}" if index == current_index and not terminal else ""
        lines.append(f"- {mark} {name}{suffix}")

    if view.get("replan"):
        lines.extend(["", f"重规划 `{view['replan']}` 次"])

    tasks = view.get("tasks", {})
    if tasks:
        completed = sum(
            task.get("status") in {"success", "failed", "timeout", "cancelled"}
            for task in tasks.values()
        )
        lines.extend(["", f"**子任务 {completed}/{len(tasks)}**"])
        for task_id, task in list(tasks.items())[:6]:
            task_status = str(task.get("status", "waiting"))
            mark = {
                "success": "OK",
                "running": "RUN",
                "failed": "ERR",
                "timeout": "TIME",
                "cancelled": "STOP",
            }.get(task_status, "WAIT")
            description = str(task.get("description", "")).replace("\n", " ")[:72]
            lines.append(f"- `{mark}` **{task_id}** · {description}")

    evidence = view.get("evidence", {})
    if evidence:
        lines.extend(
            [
                "",
                f"**证据覆盖 {float(evidence.get('coverage', 0.0) or 0.0):.1%}** · "
                f"{int(evidence.get('supported_count', 0) or 0)} supported · "
                f"{int(evidence.get('not_enough_evidence_count', 0) or 0)} NEI",
            ]
        )

    events = view.get("events", [])
    if events:
        lines.extend(["", "**最近事件**"])
        lines.extend(f"- {event}" for event in events[-6:])
    return "\n".join(lines)


def render_evidence(evidence: dict[str, Any] | None) -> str:
    if not evidence:
        return "证据审计尚未开始。"
    claims = evidence.get("claims", [])
    total = len(claims) or int(evidence.get("claim_count", 0) or 0)
    lines = [
        "### 证据审计",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| Claim 覆盖率 | **{float(evidence.get('coverage', 0.0) or 0.0):.1%}** |",
        f"| Supported | {int(evidence.get('supported_count', 0) or 0)} / {total} |",
        f"| Refuted | {int(evidence.get('refuted_count', 0) or 0)} |",
        f"| NEI | {int(evidence.get('not_enough_evidence_count', evidence.get('nei_count', 0)) or 0)} |",
        f"| 来源数 | {int(evidence.get('source_count', 0) or 0)} |",
        f"| 原始/权威来源 | {float(evidence.get('primary_source_ratio', 0.0) or 0.0):.1%} |",
        f"| 全文证据来源 | {float(evidence.get('fulltext_source_ratio', 0.0) or 0.0):.1%} |",
        f"| 缺口补充轮数 | {int(evidence.get('gap_rounds', 0) or 0)} |",
    ]
    task_coverage = evidence.get("task_coverage") or {}
    if task_coverage:
        lines.append(
            f"| 规划维度覆盖 | **{float(task_coverage.get('coverage', 0.0) or 0.0):.1%}** "
            f"({int(task_coverage.get('covered_count', 0) or 0)}/"
            f"{int(task_coverage.get('required_count', 0) or 0)}) |"
        )
    unresolved = [claim for claim in claims if claim.get("status") != "supported"]
    if unresolved:
        lines.extend(["", "#### 未决 Claim", ""])
        for claim in unresolved[:8]:
            status = str(claim.get("status", "not_enough_evidence")).upper()
            lines.append(f"- `{status}` {claim.get('text', '')}")
    sources = evidence.get("sources", [])
    if sources:
        lines.extend(["", "#### 高价值来源", ""])
        ranked = sorted(
            sources,
            key=lambda source: (
                bool(source.get("is_primary", False)),
                float(source.get("quality_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        for source in ranked[:8]:
            title = source.get("title") or source.get("publisher") or source.get("url") or "Untitled"
            url = source.get("url", "")
            quality = float(source.get("quality_score", 0.0) or 0.0)
            link = f"[{title}]({url})" if url else str(title)
            lines.append(f"- {link} · quality `{quality:.2f}`")
    revision = evidence.get("revision", {})
    if revision.get("attempted"):
        before = revision.get("before", {})
        after = revision.get("after", {})
        result = "采用" if revision.get("accepted") else "回滚"
        lines.extend(
            [
                "",
                "#### 证据约束修订",
                "",
                f"- 质量门控：**{result}**",
                f"- 覆盖率：{float(before.get('coverage', 0.0) or 0.0):.1%} → "
                f"{float(after.get('coverage', 0.0) or 0.0):.1%}",
                f"- Claim：{int(before.get('claim_count', 0) or 0)} → "
                f"{int(after.get('claim_count', 0) or 0)}",
                f"- 原因：{revision.get('reason', '')}",
            ]
        )
    return "\n".join(lines)


def history_choices(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    choices = []
    for row in rows:
        created = str(row.get("created_at", ""))
        try:
            label_time = datetime.fromisoformat(created).astimezone().strftime("%m-%d %H:%M")
        except ValueError:
            label_time = created[:16]
        status = str(row.get("status", "unknown")).upper().replace("_", " ")
        query = " ".join(str(row.get("query", "")).split())
        if len(query) > 54:
            query = query[:53] + "..."
        choices.append((f"{label_time} · {status} · {query}", str(row.get("run_id", ""))))
    return choices


def render_history_summary(row: dict[str, Any], events: list[dict[str, Any]]) -> str:
    if not row:
        return "暂无运行记录。"
    lines = [
        f"### {str(row.get('status', 'unknown')).upper().replace('_', ' ')}",
        "",
        f"**{row.get('query', '')}**",
        "",
        "| 运行指标 | 结果 |",
        "|---|---:|",
        f"| 后端 | `{row.get('backend', '')}` |",
        f"| 总耗时 | {float(row.get('elapsed_seconds', 0.0) or 0.0):.1f}s |",
        f"| 置信度 | {float(row.get('confidence', 0.0) or 0.0):.2f} |",
        f"| 证据覆盖率 | {float(row.get('coverage', 0.0) or 0.0):.1%} |",
        f"| Supported | {int(row.get('supported_count', 0) or 0)} / {int(row.get('claim_count', 0) or 0)} |",
        f"| 来源 | {int(row.get('source_count', 0) or 0)} |",
        f"| 搜索调用 | {int(row.get('num_searches', 0) or 0)} |",
        f"| 重规划 | {int(row.get('num_replan', 0) or 0)} |",
    ]
    usage = dict(row.get("metadata", {}).get("model_usage") or {})
    assignments = dict(row.get("metadata", {}).get("model_assignments") or {})
    models = sorted({str(model) for model in assignments.values() if model})
    if models:
        lines.append(f"| 模型 | `{', '.join(models)}` |")
    if usage:
        lines.extend(
            [
                f"| LLM API 调用 | {int(usage.get('api_calls', 0) or 0)} |",
                f"| Provider tokens | {int(usage.get('total_tokens', 0) or 0):,} |",
            ]
        )
    lines.extend(["", f"`{row.get('run_id', '')}`"])
    if row.get("error"):
        lines.extend(["", f"**错误**：{row['error']}"])
    if events:
        lines.extend(["", "#### 事件时间线", ""])
        for event in events[-20:]:
            stamp = datetime.fromtimestamp(float(event.get("timestamp", 0.0))).strftime("%H:%M:%S")
            label = _event_label(
                str(event.get("kind", "")),
                str(event.get("message", "")),
                dict(event.get("payload") or {}),
            )
            if label:
                lines.append(f"- `{stamp}` {label}")
    return "\n".join(lines)


__all__ = [
    "apply_run_event",
    "history_choices",
    "new_run_view",
    "render_evidence",
    "render_history_summary",
    "render_progress",
]
