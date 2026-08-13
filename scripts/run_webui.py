#!/usr/bin/env python3
"""Streaming Gradio workspace with durable runs and cooperative cancellation."""
from __future__ import annotations

import asyncio
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from src.core.runner import (
    initialize_modules,
    load_config,
    run_research_with_metadata,
    save_report,
    setup_logging,
)
from src.runtime import RunController, RunStore
from src.runtime.presentation import (
    apply_run_event,
    history_choices,
    new_run_view,
    render_evidence,
    render_history_summary,
    render_progress,
)


_MODULES = ["solver", "planner", "summarizer", "judge", "red_agent", "blue_agent", "compressor"]
_INIT_HINT = "尚无运行。"
_EVIDENCE_INIT = "证据审计尚未开始。"
_RUN_STORE = RunStore(PROJECT_ROOT / "outputs" / "runs" / "runs.db")
_RUN_CONTROLLER = RunController(_RUN_STORE)
_REPORT_ROOT = (PROJECT_ROOT / "outputs" / "reports").resolve()
_EVIDENCE_ROOT = (PROJECT_ROOT / "outputs" / "evidence").resolve()

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


def _safe_artifact_path(raw_path: str, allowed_root: Path) -> Path | None:
    """Resolve a ledger path without allowing the history UI to expose other files."""
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file() or not resolved.is_relative_to(allowed_root):
        return None
    return resolved


def _evidence_from_artifact(raw_path: str) -> dict[str, Any]:
    path = _safe_artifact_path(raw_path, _EVIDENCE_ROOT)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    audit = dict(payload.get("audit") or {})
    audit["sources"] = list(payload.get("sources") or [])
    audit["artifact"] = str(path)
    audit["revision"] = dict(payload.get("run_metadata", {}).get("evidence_revision") or {})
    return audit


def _request_cancel(run_id: str):
    requested = _RUN_CONTROLLER.cancel(run_id, "Cancelled from the Web UI.")
    if requested:
        return "`STOPPING` 当前调用结束后停止，并保留已完成结果。", gr.update(interactive=False)
    if run_id and _RUN_STORE.get_run(run_id):
        return "该运行已结束或正在停止。", gr.update(interactive=False)
    return "没有可停止的运行。", gr.update(interactive=False)


def do_research_stream(query: str, backend: str, use_adversarial: bool):
    """Yield structured progress, report, audit, artifact, and UI control state."""
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

    cfg = load_config()
    cfg.setdefault("model", {})["backend"] = backend
    cfg["model"]["backend_mapping"] = {module: backend for module in _MODULES}
    cfg.setdefault("adversarial", {})["enabled"] = bool(use_adversarial)

    handle = _RUN_CONTROLLER.create_run(
        query=query,
        backend=backend,
        adversarial=bool(use_adversarial),
    )
    holder: dict[str, Any] = {}

    def _worker() -> None:
        try:
            modules = initialize_modules(
                cfg,
                session_id=handle.run_id,
                run_id=handle.run_id,
                event_sink=_RUN_CONTROLLER.event_sink,
                cancellation_token=handle.token,
            )
            report, metadata = asyncio.run(run_research_with_metadata(query, cfg, modules))
            evidence = modules["orchestrator"].get_evidence_snapshot()
            report_path = save_report(
                report,
                query,
                output_dir=str(_REPORT_ROOT),
            )
            status = str(metadata.get("run_status") or "complete")
            _RUN_STORE.complete_run(
                handle.run_id,
                status=status,
                report_path=report_path,
                evidence_path=str(metadata.get("evidence_artifact") or ""),
                metadata=metadata,
            )
            holder.update(
                {
                    "report": report,
                    "metadata": metadata,
                    "evidence": evidence,
                    "download": report_path,
                }
            )
        except Exception as exc:  # noqa: BLE001 - all failures must reach UI and ledger
            error = f"{type(exc).__name__}: {exc}"
            holder["error"] = error
            _RUN_STORE.fail_run(handle.run_id, error)
        finally:
            _RUN_CONTROLLER.finish(handle.run_id)

    worker = threading.Thread(
        target=_worker,
        name=f"research-{handle.run_id}",
        daemon=True,
    )
    started = time.monotonic()
    view = new_run_view()
    worker.start()

    controls_running = (
        gr.update(interactive=False),
        gr.update(interactive=True),
    )
    yield (
        render_progress(
            view,
            elapsed=0.0,
            backend=backend,
            adversarial=use_adversarial,
            run_id=handle.run_id,
        ),
        "",
        _EVIDENCE_INIT,
        None,
        handle.run_id,
        *controls_running,
        "",
    )

    while worker.is_alive() or not handle.events.empty():
        changed = False
        while True:
            try:
                event = handle.events.get_nowait()
            except queue.Empty:
                break
            apply_run_event(view, event)
            changed = True
        if handle.token.is_cancelled and view.get("status") == "running":
            view["status"] = "cancelling"
            changed = True

        elapsed = time.monotonic() - started
        if changed or worker.is_alive():
            controls_current = (
                gr.update(interactive=False),
                gr.update(interactive=not handle.token.is_cancelled),
            )
            yield (
                render_progress(
                    view,
                    elapsed=elapsed,
                    backend=backend,
                    adversarial=use_adversarial,
                    run_id=handle.run_id,
                ),
                "",
                render_evidence(view.get("evidence")),
                None,
                handle.run_id,
                *controls_current,
                (
                    "`STOPPING` 当前调用结束后停止，并保留已完成结果。"
                    if handle.token.is_cancelled
                    else ""
                ),
            )
        if worker.is_alive():
            time.sleep(0.35)

    worker.join(timeout=1.0)
    elapsed = time.monotonic() - started
    controls_done = (
        gr.update(interactive=True),
        gr.update(interactive=False),
    )
    if holder.get("error"):
        view["status"] = "failed"
        yield (
            render_progress(
                view,
                elapsed=elapsed,
                backend=backend,
                adversarial=use_adversarial,
                run_id=handle.run_id,
            ),
            "",
            render_evidence(view.get("evidence")),
            None,
            handle.run_id,
            *controls_done,
            f"**运行失败**：{holder['error']}",
        )
        return

    metadata = dict(holder.get("metadata") or {})
    view["status"] = str(metadata.get("run_status") or view.get("status") or "complete")
    final_evidence = holder.get("evidence")
    if isinstance(final_evidence, dict) and final_evidence:
        view["evidence"] = final_evidence
    yield (
        render_progress(
            view,
            elapsed=elapsed,
            backend=backend,
            adversarial=use_adversarial,
            run_id=handle.run_id,
        ),
        str(holder.get("report") or "（无报告）"),
        render_evidence(view.get("evidence")),
        holder.get("download"),
        handle.run_id,
        *controls_done,
        "",
    )


def _load_history_run(run_id: str):
    row = _RUN_STORE.get_run(str(run_id or ""))
    if not row:
        return "暂无运行记录。", "", _EVIDENCE_INIT, None
    events = _RUN_STORE.get_events(row["run_id"])
    summary = render_history_summary(row, events)
    report_path = _safe_artifact_path(str(row.get("report_path") or ""), _REPORT_ROOT)
    try:
        report = report_path.read_text(encoding="utf-8") if report_path else ""
    except OSError:
        report = ""
    evidence = _evidence_from_artifact(str(row.get("evidence_path") or ""))
    if not evidence:
        evidence = dict(row.get("metadata", {}).get("evidence_summary") or {})
        evidence["source_count"] = int(row.get("source_count", 0) or 0)
        evidence["revision"] = dict(
            row.get("metadata", {}).get("evidence_revision") or {}
        )
    return summary, report, render_evidence(evidence), str(report_path) if report_path else None


def _refresh_history(selected_run_id: str = ""):
    rows = _RUN_STORE.list_runs(limit=50)
    choices = history_choices(rows)
    available = {value for _, value in choices}
    selected = selected_run_id if selected_run_id in available else (choices[0][1] if choices else None)
    summary, report, evidence, download = _load_history_run(selected or "")
    return gr.update(choices=choices, value=selected), summary, report, evidence, download


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="YuResearchAgent") as demo:
        active_run_id = gr.State("")
        gr.Markdown(
            "# YuResearchAgent\nEvidence-grounded multi-agent research workspace",
            elem_classes=["app-header"],
        )

        with gr.Tabs():
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
    recovered = _RUN_STORE.recover_interrupted()
    if recovered:
        print(f"Recovered {recovered} interrupted run(s) in the local ledger.")
    build_ui().queue(default_concurrency_limit=4).launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="gray"),
        css=_APP_CSS,
    )
