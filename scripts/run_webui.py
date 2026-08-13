#!/usr/bin/env python3
"""Gradio adapter for the research runtime, history, and verified demos."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from src.core.runner import setup_logging
from src.runtime.presentation import (
    history_choices,
    render_evidence,
    render_history_summary,
    render_progress,
)
from src.web import DemoArtifact, DemoCatalog, ResearchRunService

_INIT_HINT = "尚无运行。"
_EVIDENCE_INIT = "证据审计尚未开始。"
_SERVICE = ResearchRunService(PROJECT_ROOT)
_DEMOS = DemoCatalog(
    PROJECT_ROOT,
    "docs/evaluation/demos/catalog.json",
)

_APP_CSS = """
.gradio-container { max-width: 1480px !important; }
.app-header h1 { font-size: 24px !important; margin: 0 0 2px 0 !important; letter-spacing: 0 !important; }
.app-header p { color: var(--body-text-color-subdued); margin: 0 !important; }
.run-column { min-height: 560px; }
button, textarea, input { border-radius: 6px !important; }
.compact-action { min-width: 112px !important; }
@media (max-width: 700px) {
  .query-controls, .action-controls, .history-controls {
    flex-direction: column !important;
    align-items: stretch !important;
  }
  .query-controls > *, .action-controls > *, .history-controls > * {
    width: 100% !important;
    min-width: 0 !important;
    flex: 1 1 auto !important;
  }
  .run-column { min-height: 0; }
  .action-controls .compact-action, .history-controls .compact-action {
    min-height: 40px;
  }
}
"""


def _request_cancel(run_id: str):
    outcome = _SERVICE.request_cancel(run_id)
    if outcome == "requested":
        return "`STOPPING` 当前调用结束后停止，并保留已完成结果。", gr.update(interactive=False)
    if outcome == "terminal":
        return "该运行已结束或正在停止。", gr.update(interactive=False)
    return "没有可停止的运行。", gr.update(interactive=False)


def do_research_stream(query: str, backend: str, use_adversarial: bool):
    """Project framework-neutral run updates into Gradio component values."""
    query = str(query or "").strip()
    if not query:
        yield (
            "请输入研究问题。",
            "",
            _EVIDENCE_INIT,
            None,
            "",
            gr.update(interactive=True),
            gr.update(interactive=False),
            "",
        )
        return

    for update in _SERVICE.stream(query, backend, bool(use_adversarial)):
        status = str(update.view.get("status") or "running")
        stopping = status == "cancelling"
        notice = ""
        if stopping:
            notice = "`STOPPING` 当前调用结束后停止，并保留已完成结果。"
        if update.error:
            notice = f"**运行失败**：{update.error}"
        yield (
            render_progress(
                update.view,
                elapsed=update.elapsed_seconds,
                backend=update.backend,
                adversarial=update.adversarial,
                run_id=update.run_id,
            ),
            update.report,
            render_evidence(update.evidence),
            update.download_path,
            update.run_id,
            gr.update(interactive=update.terminal),
            gr.update(interactive=not update.terminal and not stopping),
            notice,
        )


def _load_history_run(run_id: str):
    artifact = _SERVICE.load_history(str(run_id or ""))
    if artifact is None:
        return "暂无运行记录。", "", _EVIDENCE_INIT, None
    return (
        render_history_summary(artifact.row, artifact.events),
        artifact.report,
        render_evidence(artifact.evidence),
        artifact.download_path,
    )


def _refresh_history(selected_run_id: str = ""):
    choices = history_choices(_SERVICE.list_history(limit=50))
    available = {value for _, value in choices}
    selected = selected_run_id if selected_run_id in available else (choices[0][1] if choices else None)
    summary, report, evidence, download = _load_history_run(selected or "")
    return gr.update(choices=choices, value=selected), summary, report, evidence, download


def _render_demo_summary(demo: DemoArtifact) -> str:
    runtime = demo.runtime
    return "\n".join(
        [
            "### VERIFIED · COMPLETE",
            "",
            f"**{demo.query}**",
            "",
            "| 运行指标 | 结果 |",
            "|---|---:|",
            f"| 后端 / 模型 | `{runtime.get('backend', '')}` / `{runtime.get('model', '')}` |",
            f"| 总耗时 | {float(runtime.get('elapsed_seconds', 0.0) or 0.0):.1f}s |",
            f"| LLM API 调用 | {int(runtime.get('api_calls', 0) or 0)} |",
            f"| Provider tokens | {int(runtime.get('total_tokens', 0) or 0):,} |",
            f"| 搜索调用 | {int(runtime.get('search_calls', 0) or 0)} |",
            f"| 来源 / 证据块 | {int(runtime.get('source_count', 0) or 0)} / "
            f"{int(runtime.get('evidence_count', 0) or 0)} |",
            f"| 原始来源 | {float(runtime.get('primary_source_ratio', 0.0) or 0.0):.1%} |",
            f"| 全文来源 | {float(runtime.get('fulltext_source_ratio', 0.0) or 0.0):.1%} |",
            "",
            "`SHA-256 VERIFIED`",
            "",
            demo.interpretation,
        ]
    )


def _load_demo(demo_id: str):
    demo = _DEMOS.load(str(demo_id or _DEMOS.default_id))
    return (
        _render_demo_summary(demo),
        demo.report,
        render_evidence(demo.evidence),
        demo.report_path,
    )


def build_ui() -> gr.Blocks:
    default_demo = _DEMOS.load(_DEMOS.default_id)
    with gr.Blocks(title="YuResearchAgent") as demo:
        active_run_id = gr.State("")
        gr.Markdown(
            "# YuResearchAgent\nEvidence-grounded multi-agent research workspace",
            elem_classes=["app-header"],
        )

        with gr.Tabs():
            with gr.Tab("验证样例"):
                demo_select = gr.Dropdown(
                    choices=_DEMOS.choices(),
                    value=_DEMOS.default_id,
                    label="不可变运行工件",
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=300):
                        demo_summary = gr.Markdown(value=_render_demo_summary(default_demo))
                    with gr.Column(scale=5):
                        with gr.Tabs():
                            with gr.Tab("样例报告"):
                                demo_report = gr.Markdown(value=default_demo.report)
                            with gr.Tab("证据复核"):
                                demo_evidence = gr.Markdown(value=render_evidence(default_demo.evidence))
                demo_download = gr.DownloadButton(
                    "下载已验证报告",
                    value=default_demo.report_path,
                    elem_classes=["compact-action"],
                )

            with gr.Tab("研究工作台"):
                with gr.Row(elem_classes=["query-controls"]):
                    query = gr.Textbox(
                        label="研究问题",
                        placeholder="例如：比较近两年深度研究 Agent 的证据归因方法",
                        lines=2,
                        scale=5,
                    )
                    backend = gr.Dropdown(
                        choices=["kimi", "qwen", "deepseek", "glm"],
                        value="kimi",
                        label="模型后端",
                        scale=1,
                        min_width=150,
                    )
                    use_adv = gr.Checkbox(
                        value=False,
                        label="对抗精修",
                        scale=1,
                        min_width=130,
                    )
                with gr.Row(elem_classes=["action-controls"]):
                    start_button = gr.Button(
                        "开始研究",
                        variant="primary",
                        elem_classes=["compact-action"],
                    )
                    stop_button = gr.Button(
                        "停止运行",
                        variant="stop",
                        interactive=False,
                        elem_classes=["compact-action"],
                    )
                    stop_notice = gr.Markdown(value="")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=300, elem_classes=["run-column"]):
                        gr.Markdown("#### 运行状态")
                        progress = gr.Markdown(value=_INIT_HINT)
                    with gr.Column(scale=5, elem_classes=["run-column"]):
                        with gr.Tabs():
                            with gr.Tab("研究报告"):
                                report = gr.Markdown(value="")
                            with gr.Tab("证据审计"):
                                evidence = gr.Markdown(value=_EVIDENCE_INIT)
                download = gr.DownloadButton(
                    "下载 Markdown 报告",
                    value=None,
                    elem_classes=["compact-action"],
                )

            with gr.Tab("运行历史"):
                with gr.Row(elem_classes=["history-controls"]):
                    history_select = gr.Dropdown(
                        choices=[],
                        label="运行记录",
                        scale=5,
                    )
                    refresh_button = gr.Button(
                        "刷新记录",
                        scale=1,
                        elem_classes=["compact-action"],
                    )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=300):
                        history_summary = gr.Markdown(value="暂无运行记录。")
                    with gr.Column(scale=5):
                        with gr.Tabs():
                            with gr.Tab("历史报告"):
                                history_report = gr.Markdown(value="")
                            with gr.Tab("历史证据"):
                                history_evidence = gr.Markdown(value=_EVIDENCE_INIT)
                history_download = gr.DownloadButton(
                    "下载历史报告",
                    value=None,
                    elem_classes=["compact-action"],
                )

        demo_select.change(
            _load_demo,
            inputs=[demo_select],
            outputs=[demo_summary, demo_report, demo_evidence, demo_download],
            show_progress="hidden",
        )
        research_event = start_button.click(
            do_research_stream,
            inputs=[query, backend, use_adv],
            outputs=[
                progress,
                report,
                evidence,
                download,
                active_run_id,
                start_button,
                stop_button,
                stop_notice,
            ],
            show_progress="hidden",
            concurrency_limit=1,
            concurrency_id="research-runs",
        )
        stop_button.click(
            _request_cancel,
            inputs=[active_run_id],
            outputs=[stop_notice, stop_button],
            queue=False,
        )
        research_event.then(
            _refresh_history,
            inputs=[history_select],
            outputs=[
                history_select,
                history_summary,
                history_report,
                history_evidence,
                history_download,
            ],
            show_progress="hidden",
        )
        history_select.change(
            _load_history_run,
            inputs=[history_select],
            outputs=[history_summary, history_report, history_evidence, history_download],
            show_progress="hidden",
        )
        refresh_button.click(
            _refresh_history,
            inputs=[history_select],
            outputs=[
                history_select,
                history_summary,
                history_report,
                history_evidence,
                history_download,
            ],
            show_progress="hidden",
        )
        demo.load(
            _refresh_history,
            inputs=[history_select],
            outputs=[
                history_select,
                history_summary,
                history_report,
                history_evidence,
                history_download,
            ],
            show_progress="hidden",
        )
    return demo


if __name__ == "__main__":
    setup_logging("INFO")
    recovered = _SERVICE.recover_interrupted()
    if recovered:
        print(f"Recovered {recovered} interrupted run(s) in the local ledger.")
    build_ui().queue(default_concurrency_limit=4).launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="gray"),
        css=_APP_CSS,
    )
