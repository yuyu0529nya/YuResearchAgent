from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.runtime import RunEvent
from src.web import DemoCatalog, ResearchRunService
from src.web.artifacts import ArtifactIntegrityError

ROOT = Path(__file__).resolve().parents[2]


def test_committed_demo_loads_with_verified_artifacts() -> None:
    catalog = DemoCatalog(ROOT, "docs/evaluation/demos/catalog.json")

    demo = catalog.load(catalog.default_id)

    assert demo.report.startswith("#")
    assert demo.runtime["model"] == "k3"
    assert demo.evidence["sources"]
    assert Path(demo.report_path).is_file()


def test_demo_catalog_rejects_hash_tampering(tmp_path: Path) -> None:
    payload = json.loads((ROOT / "docs/evaluation/demos/catalog.json").read_text(encoding="utf-8"))
    payload["demos"][0]["report"]["sha256"] = "0" * 64
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    catalog = DemoCatalog(ROOT, catalog_path)

    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        catalog.load(catalog.default_id)


def test_research_service_streams_and_persists_without_gradio(tmp_path: Path) -> None:
    class _Orchestrator:
        @staticmethod
        def get_evidence_snapshot():
            return {"coverage": 1.0, "claims": []}

    def initialize(_config, *, run_id, event_sink, **_kwargs):
        event_sink(
            RunEvent(
                run_id=run_id,
                sequence=1,
                kind="run_started",
                state="planning",
            )
        )
        return {"orchestrator": _Orchestrator()}

    async def run(query, _config, _modules):
        return f"# {query}", {
            "run_status": "complete",
            "elapsed_seconds": 0.01,
            "source_count": 0,
            "evidence_audit": {"coverage": 1.0, "claims": []},
        }

    def save(report, _query, output_dir):
        path = Path(output_dir) / "report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        return str(path)

    service = ResearchRunService(
        tmp_path,
        config_loader=lambda: {"model": {}, "adversarial": {}},
        module_initializer=initialize,
        research_runner=run,
        report_saver=save,
        poll_interval=0.01,
    )

    updates = list(service.stream("test query", "kimi", False))

    assert updates[-1].terminal is True
    assert updates[-1].report == "# test query"
    assert updates[-1].evidence["coverage"] == 1.0
    row = service.list_history()[0]
    assert row["status"] == "complete"
    history = service.load_history(row["run_id"])
    assert history is not None
    assert history.report == "# test query"
