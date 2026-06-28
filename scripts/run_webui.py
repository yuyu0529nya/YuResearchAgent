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
  - 可切换模型后端（qwen / deepseek / glm）
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

    head = f"**⏱ {elapsed:.0f}s** · 后端 `{backend}`"
    if replan:
        head += f" · 重规划 ×{replan}"
    lines = [head, ""]

    if cur_idx == -1 and not done:
        lines.append("- 🔄 **初始化** — 加载模型与工具…")

    for i, st in enumerate(pipeline):
        name, desc = _STAGE_MAP[st]
        if done or i < cur_idx:
            mark = "✅"
        elif i == cur_idx:
            mark = "🔄"
        else:
            mark = "⏳"
        suffix = f" — {desc}" if (i == cur_idx and not done) else ""
        lines.append(f"- {mark} {name}{suffix}")

    if done:
        lines.append("\n**✅ 研究完成**")
    return "\n".join(lines)


def do_research_stream(query: str, backend: str, use_adversarial: bool):
    """Gradio 流式生成器：yield (进度面板 Markdown, 报告 Markdown)。"""
    if not query or not query.strip():
        yield "⚠️ 请输入研究问题。", ""
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

    holder: dict[str, str] = {}

    def _worker():
        try:
            modules = initialize_modules(cfg, session_id="webui")
            holder["report"] = asyncio.run(run_research(query, cfg, modules))
        except Exception as e:  # noqa: BLE001  — 任何失败都回传 UI，不让线程静默死掉
            holder["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=_worker, daemon=True)
    t0 = time.time()
    t.start()

    current: str | None = None
    replan = 0
    try:
        while t.is_alive() or not log_q.empty():
            try:
                msg = log_q.get(timeout=0.4)
            except queue.Empty:
                # 心跳：刷新活时长，让用户看到仍在运行
                yield _render_progress(current, time.time() - t0, backend, use_adversarial, replan=replan), ""
                continue
            if "State transition:" in msg:
                state = msg.rsplit(":", 1)[-1].strip()
                if state == "replanning":
                    replan += 1
                if state in _STAGE_MAP or state == "replanning":
                    current = state
                    yield _render_progress(current, time.time() - t0, backend, use_adversarial, replan=replan), ""
        t.join(timeout=5)
    finally:
        for lg in loggers:
            lg.removeHandler(handler)

    elapsed = time.time() - t0
    if "error" in holder:
        panel = _render_progress(current, elapsed, backend, use_adversarial, replan=replan)
        yield panel + f"\n\n❌ **出错**：{holder['error']}", ""
        return

    report = holder.get("report", "（无结果，请重试）")
    yield _render_progress(current, elapsed, backend, use_adversarial, done=True, replan=replan), report


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="YuResearchAgent", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🚀 YuResearchAgent\n"
            "### 从复杂 Query 到结构化深度研究报告，全链路自动化\n"
            "_自研编排引擎(9 状态机 + DAG 并发) · 多源检索 · Red-Blue 对抗降噪 · 多后端路由_"
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
                    choices=["qwen", "deepseek", "glm"],
                    value="qwen",
                    label="模型后端",
                )
                use_adv = gr.Checkbox(value=False, label="启用对抗精修（更慢）")
        btn = gr.Button("🔍 开始研究", variant="primary")
        gr.Markdown("> 完整研究约需 2–4 分钟（规划 → 并发检索 → 合成）。进度见下方面板。")

        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("#### 实时进度")
                progress = gr.Markdown(value=_INIT_HINT)
            with gr.Column(scale=2):
                gr.Markdown("#### 研究报告")
                report = gr.Markdown(value="")

        btn.click(
            do_research_stream,
            inputs=[query, backend, use_adv],
            outputs=[progress, report],
        )

        gr.Examples(
            examples=[
                ["2024 年大语言模型 Agent 的关键技术趋势与代表性框架", "qwen", False],
                ["对比分析 GPT-4o、Claude 3.5、Gemini 1.5 的能力差异", "deepseek", False],
            ],
            inputs=[query, backend, use_adv],
        )
    return demo


if __name__ == "__main__":
    setup_logging("INFO")
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
