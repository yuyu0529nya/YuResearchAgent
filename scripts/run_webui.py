#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_webui.py
================================================================================
YuResearchAgent 的 Gradio Web 界面。

特性：
  - 输入研究问题 → 跑完整多智能体流程 → 流式展示结构化报告
  - **实时进度面板**：把编排器的状态机转移（planning → dispatching → … → done）
    通过日志流实时映射成管线 checklist + 活时长计时，长任务不再"干等黑屏"
  - 可切换模型后端（Kimi / Qwen / DeepSeek / GLM）
  - 可一键开关 Red-Blue 对抗精修

实现要点：研究流程在后台线程跑（asyncio.run），主生成器从一个挂在
`orchestrator` / `runner` logger 上的队列 Handler 读取进度并 yield，
无需侵入式改动编排器即可拿到状态机进度。

用法：
    pip install gradio
    python scripts/run_webui.py        # 打开 http://localhost:7860
================================================================================
"""
from __future__ import annotations

import asyncio
import logging
import queue
import re
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from src.core.runner import initialize_modules, load_config, run_research, setup_logging

_MODULES = ["solver", "planner", "summarizer", "judge", "red_agent", "blue_agent", "compressor"]

# 状态机状态 → (中文阶段名, 当前阶段说明)
_STAGE_MAP = {
    "planning": ("规划", "拆解问题为子任务 DAG"),
    "dispatching": ("检索", "并发执行子任务 · 多源搜索"),
    "collecting": ("汇总", "收集各子任务结果"),
    "synthesizing": ("合成", "撰写结构化报告"),
    "adversarial": ("对抗精修", "Red-Blue 降噪"),
}
_INIT_HINT = "点击「开始研究」后，这里实时显示编排器状态机进度。"
_EVIDENCE_INIT = "证据审计尚未开始。"
_APP_CSS = """
.gradio-container { max-width: 1440px !important; }
.app-header h1 { font-size: 24px !important; margin-bottom: 2px !important; }
.app-header p { color: var(--body-text-color-subdued); margin-top: 0 !important; }
.panel { border-radius: 6px !important; }
button { border-radius: 6px !important; }
textarea, input { border-radius: 6px !important; }
"""


class _QueueLogHandler(logging.Handler):
    """把日志记录推进队列，供前台生成器读取做进度展示。"""

    def __init__(self, q: "queue.Queue[str]"):
        super().__init__()
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._q.put_nowait(record.getMessage())
        except Exception:
            pass


def _render_progress(
    current: str | None,
    elapsed: float,
    backend: str,
    adversarial: bool,
    *,
    done: bool = False,
    replan: int = 0,
    evidence: dict | None = None,
    events: list[str] | None = None,
) -> str:
    """把当前状态渲染成 Markdown 进度面板。"""
    pipeline = ["planning", "dispatching", "collecting", "synthesizing"]
    if adversarial:
        pipeline.append("adversarial")

    if done:
        cur_idx = len(pipeline)
    elif current == "replanning":
        cur_idx = pipeline.index("dispatching")  # 重规划回环到检索阶段
    elif current in pipeline:
        cur_idx = pipeline.index(current)
    else:
        cur_idx = -1  # 尚在初始化

    head = f"**{elapsed:.0f}s** · 后端 `{backend}`"
    if replan:
        head += f" · 重规划 ×{replan}"
    lines = [head, ""]

    if cur_idx == -1 and not done:
        lines.append("- `RUN` **初始化** - 加载模型与工具")

    for i, st in enumerate(pipeline):
        name, desc = _STAGE_MAP[st]
        if done or i < cur_idx:
            mark = "`OK`"
        elif i == cur_idx:
            mark = "`RUN`"
        else:
            mark = "`WAIT`"
        suffix = f" - {desc}" if (i == cur_idx and not done) else ""
        lines.append(f"- {mark} {name}{suffix}")

    if evidence:
        lines.extend(
            [
                "",
                f"**证据覆盖 {evidence.get('coverage', 0.0):.1%}** · "
                f"{evidence.get('supported_count', 0)} supported · "
                f"{evidence.get('not_enough_evidence_count', evidence.get('nei_count', 0))} NEI",
            ]
        )
    if events:
        lines.extend(["", "**最近事件**"])
        lines.extend(f"- {event}" for event in events[-5:])
    if done:
        lines.append("\n**研究完成**")
    return "\n".join(lines)


def _update_evidence_from_log(message: str, evidence: dict) -> None:
    match = re.search(
        r"coverage=([\d.]+)%.*claims=(\d+).*supported=(\d+).*NEI=(\d+).*sources=(\d+).*"
        r"fulltext=([\d.]+)%.*primary=([\d.]+)%",
        message,
    )
    if match:
        coverage, claims, supported, nei, sources, fulltext, primary = match.groups()
        evidence.update(
            {
                "coverage": float(coverage) / 100,
                "claim_count": int(claims),
                "supported_count": int(supported),
                "not_enough_evidence_count": int(nei),
                "source_count": int(sources),
                "fulltext_source_ratio": float(fulltext) / 100,
                "primary_source_ratio": float(primary) / 100,
            }
        )


def _event_from_log(message: str) -> str | None:
    if "DAG 生成完成:" in message:
        return message.split("DAG 生成完成:", 1)[1].strip()
    if message.startswith("[Dispatch] ▶"):
        return message.replace("[Dispatch] ▶ ", "", 1)
    if message.startswith("[Collect]"):
        return message.split("]", 1)[1].strip()
    if message.startswith("[Evidence]"):
        return message.split("]", 1)[1].strip()
    if message.startswith("[Replan]"):
        return message.split("]", 1)[1].strip()
    return None


def _render_evidence(evidence: dict | None) -> str:
    if not evidence:
        return _EVIDENCE_INIT
    claims = evidence.get("claims", [])
    total = len(claims) or evidence.get("claim_count", 0)
    lines = [
        "### 证据审计",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| Claim 覆盖率 | **{evidence.get('coverage', 0.0):.1%}** |",
        f"| Supported | {evidence.get('supported_count', 0)} / {total} |",
        f"| Refuted | {evidence.get('refuted_count', 0)} |",
        f"| NEI | {evidence.get('not_enough_evidence_count', evidence.get('nei_count', 0))} |",
        f"| 来源数 | {evidence.get('source_count', 0)} |",
        f"| 原始/权威来源 | {evidence.get('primary_source_ratio', 0.0):.1%} |",
        f"| 全文证据来源 | {evidence.get('fulltext_source_ratio', 0.0):.1%} |",
        f"| 缺口补充轮数 | {evidence.get('gap_rounds', 0)} |",
    ]
    unresolved = [claim for claim in claims if claim.get("status") != "supported"]
    if unresolved:
        lines.extend(["", "#### 未决 Claim", ""])
        for claim in unresolved[:8]:
            status = claim.get("status", "not_enough_evidence").upper()
            lines.append(f"- `{status}` {claim.get('text', '')}")
    sources = evidence.get("sources", [])
    if sources:
        lines.extend(["", "#### 高价值来源", ""])
        ranked = sorted(
            sources,
            key=lambda source: (source.get("is_primary", False), source.get("quality_score", 0.0)),
            reverse=True,
        )
        for source in ranked[:8]:
            title = source.get("title") or source.get("publisher") or source.get("url") or "Untitled"
            url = source.get("url", "")
            quality = source.get("quality_score", 0.0)
            link = f"[{title}]({url})" if url else title
            lines.append(f"- {link} · quality `{quality:.2f}`")
    if evidence.get("artifact"):
        lines.extend(["", f"审计文件：`{evidence['artifact']}`"])
    return "\n".join(lines)


def do_research_stream(query: str, backend: str, use_adversarial: bool):
    """Gradio 流式生成器：yield (进度, 报告, 证据审计)。"""
    if not query or not query.strip():
        yield "请输入研究问题。", "", _EVIDENCE_INIT
        return

    cfg = load_config()
    cfg.setdefault("model", {})["backend"] = backend
    cfg["model"]["backend_mapping"] = {m: backend for m in _MODULES}
    cfg.setdefault("adversarial", {})["enabled"] = bool(use_adversarial)

    log_q: "queue.Queue[str]" = queue.Queue()
    handler = _QueueLogHandler(log_q)
    loggers = [logging.getLogger("orchestrator"), logging.getLogger("runner")]
    for lg in loggers:
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)

    holder: dict[str, object] = {}

    def _worker():
        try:
            modules = initialize_modules(cfg, session_id=f"webui_{time.time_ns()}")
            holder["report"] = asyncio.run(run_research(query, cfg, modules))
            holder["evidence"] = modules["orchestrator"].get_evidence_snapshot()
        except Exception as e:  # noqa: BLE001  — 任何失败都回传 UI，不让线程静默死掉
            holder["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=_worker, daemon=True)
    t0 = time.time()
    t.start()

    current: str | None = None
    replan = 0
    evidence_state: dict = {}
    events: list[str] = []
    try:
        while t.is_alive() or not log_q.empty():
            try:
                msg = log_q.get(timeout=0.4)
            except queue.Empty:
                # 心跳：刷新活时长，让用户看到仍在运行
                yield (
                    _render_progress(
                        current,
                        time.time() - t0,
                        backend,
                        use_adversarial,
                        replan=replan,
                        evidence=evidence_state,
                        events=events,
                    ),
                    "",
                    _render_evidence(evidence_state),
                )
                continue
            _update_evidence_from_log(msg, evidence_state)
            event = _event_from_log(msg)
            if event and (not events or events[-1] != event):
                events.append(event)
            if "State transition:" in msg:
                state = msg.rsplit(":", 1)[-1].strip()
                if state == "replanning":
                    replan += 1
                if state in _STAGE_MAP or state == "replanning":
                    current = state
                    yield (
                        _render_progress(
                            current,
                            time.time() - t0,
                            backend,
                            use_adversarial,
                            replan=replan,
                            evidence=evidence_state,
                            events=events,
                        ),
                        "",
                        _render_evidence(evidence_state),
                    )
        t.join(timeout=5)
    finally:
        for lg in loggers:
            lg.removeHandler(handler)

    elapsed = time.time() - t0
    if "error" in holder:
        panel = _render_progress(current, elapsed, backend, use_adversarial, replan=replan)
        yield panel + f"\n\n**出错**：{holder['error']}", "", _render_evidence(evidence_state)
        return

    report = str(holder.get("report", "（无结果，请重试）"))
    final_evidence = holder.get("evidence")
    if isinstance(final_evidence, dict) and final_evidence:
        evidence_state = final_evidence
    yield (
        _render_progress(
            current,
            elapsed,
            backend,
            use_adversarial,
            done=True,
            replan=replan,
            evidence=evidence_state,
            events=events,
        ),
        report,
        _render_evidence(evidence_state),
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="YuResearchAgent") as demo:
        gr.Markdown(
            "# YuResearchAgent\n"
            "Evidence-grounded multi-agent research workspace",
            elem_classes=["app-header"],
        )
        with gr.Row():
            query = gr.Textbox(
                label="研究问题",
                placeholder="例如：2024 年大语言模型 Agent 的关键技术趋势与代表性框架",
                lines=2,
                scale=4,
            )
            with gr.Column(scale=1, min_width=160):
                backend = gr.Dropdown(
                    choices=["kimi", "qwen", "deepseek", "glm"],
                    value="kimi",
                    label="模型后端",
                )
                use_adv = gr.Checkbox(value=False, label="启用对抗精修（更慢）")
        btn = gr.Button("开始研究", variant="primary")

        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("#### 实时进度")
                progress = gr.Markdown(value=_INIT_HINT)
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("研究报告"):
                        report = gr.Markdown(value="")
                    with gr.Tab("证据审计"):
                        evidence = gr.Markdown(value=_EVIDENCE_INIT)

        btn.click(
            do_research_stream,
            inputs=[query, backend, use_adv],
            outputs=[progress, report, evidence],
        )

    return demo


if __name__ == "__main__":
    setup_logging("INFO")
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
        css=_APP_CSS,
    )
