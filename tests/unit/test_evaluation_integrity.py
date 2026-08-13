from __future__ import annotations

from evaluation.metrics.rule_based import RuleBasedMetrics
from evaluation.benchmarks.research_bench import (
    RESEARCHBENCH_METRIC_WEIGHTS,
    ResearchBench,
)
from evaluation.evidence_metrics import evidence_quality_metrics
from evaluation.protocol import (
    load_report_artifact,
    normalize_counterbalanced_judgments,
    paired_summary,
    save_report_artifact,
)
from evaluation.report_sampling import balanced_report_excerpt
from src.core.ablation import AblationStudy
from src.core.judge import LLMJudge


def test_composite_score_uses_legacy_factual_accuracy_parts() -> None:
    metrics = {
        "factual_accuracy_str": 0.5,
        "factual_accuracy_sem": 1.0,
        "logical_consistency": 0.0,
        "citation_coverage": 0.0,
        "bias": 0.0,
        "comprehensiveness": 0.0,
    }

    assert RuleBasedMetrics.composite_score(metrics) == 0.2


def test_explicit_factual_accuracy_takes_precedence() -> None:
    metrics = {
        "factual_accuracy": 0.4,
        "factual_accuracy_str": 1.0,
        "factual_accuracy_sem": 1.0,
    }

    assert RuleBasedMetrics.composite_score(metrics, {"factual_accuracy": 1.0}) == 0.4


def test_balanced_excerpt_keeps_beginning_middle_and_bibliography() -> None:
    report = "BEGIN\n" + "A" * 5000 + "\nMIDDLE\n" + "B" * 5000 + "\n## References\nEND"

    excerpt = balanced_report_excerpt(report, max_chars=2000, segments=4)

    assert len(excerpt) <= 2000
    assert "BEGIN" in excerpt
    assert "MIDDLE" in excerpt
    assert "## References" in excerpt
    assert "END" in excerpt
    assert "omitted" in excerpt


def test_ablation_score_extraction_requires_real_evaluator_output() -> None:
    assert AblationStudy._extract_score(0.75) == 0.75
    assert AblationStudy._extract_score({"composite_score": 0.8}) == 0.8


def test_research_bench_emits_canonical_factual_accuracy(monkeypatch) -> None:
    monkeypatch.setattr(RuleBasedMetrics, "fact_accuracy", staticmethod(lambda *_: 0.5))
    monkeypatch.setattr(RuleBasedMetrics, "semantic_fact_accuracy", staticmethod(lambda *_, **__: 1.0))
    bench = ResearchBench()

    result = bench.evaluate_report("report", "tech_001")

    assert result["metrics"]["factual_accuracy"] == 0.8
    assert result["composite_score"] > 0.0
    assert result["metric_weights"] == RESEARCHBENCH_METRIC_WEIGHTS


def test_paired_summary_excludes_incomplete_pairs() -> None:
    rows = [
        {"left": {"score": 0.8}, "right": {"score": 0.5}},
        {"left": {"score": 0.7}, "right": {"error": "failed"}},
    ]

    result = paired_summary(rows, left_path="left.score", right_path="right.score", seed=7)

    assert result["n_pairs"] == 1
    assert result["left_mean"] == 0.8
    assert result["right_mean"] == 0.5


def test_counterbalanced_judge_scores_map_back_to_systems() -> None:
    result = normalize_counterbalanced_judgments(
        [
            {
                "order": {"A": "agent", "B": "baseline"},
                "result": {"accuracy": {"A": 5, "B": 3}},
            },
            {
                "order": {"A": "baseline", "B": "agent"},
                "result": {"accuracy": {"A": 2, "B": 4}},
            },
        ]
    )

    assert result["dimensions"]["accuracy"] == {
        "agent": 4.5,
        "baseline": 2.5,
        "delta": 2.0,
    }
    assert result["orders_completed"] == 2


def test_evidence_metrics_separate_citation_and_support() -> None:
    audit = {
        "claims": [
            {
                "status": "supported",
                "cited_indices": [1],
                "support_evidence_ids": ["e1"],
                "source_ids": ["s1"],
            },
            {
                "status": "not_enough_evidence",
                "cited_indices": [2],
                "support_evidence_ids": [],
                "source_ids": ["s2"],
            },
            {
                "status": "supported",
                "cited_indices": [],
                "support_evidence_ids": ["e3"],
                "source_ids": ["s3"],
            },
        ],
        "source_count": 3,
        "evidence_count": 4,
        "primary_source_ratio": 1 / 3,
        "fulltext_source_ratio": 2 / 3,
    }

    metrics = evidence_quality_metrics(audit)

    assert metrics["claim_support_coverage"] == 0.6667
    assert metrics["claim_citation_rate"] == 0.6667
    assert metrics["cited_claim_support_precision"] == 0.5
    assert metrics["supported_claim_citation_rate"] == 0.5
    assert metrics["attribution_coverage"] == 0.6667


def test_retained_report_hash_is_verified(tmp_path) -> None:
    metadata = save_report_artifact(
        "original report",
        reports_dir=tmp_path,
        question_id="q1",
        system_name="agent",
    )

    assert load_report_artifact(metadata) == "original report"
    (tmp_path / "q1_agent.md").write_text("tampered", encoding="utf-8")

    try:
        load_report_artifact(metadata)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered report should fail integrity verification")


def test_judge_comparison_validation_rejects_out_of_range_scores() -> None:
    valid = {
        dimension: {"A": 4, "B": 3, "reason": "ok"}
        for dimension in ("comprehensiveness", "accuracy", "structure", "sources")
    }
    assert LLMJudge._validate_comparison_result(valid) is not None

    invalid = dict(valid)
    invalid["accuracy"] = {"A": 8, "B": 3}
    assert LLMJudge._validate_comparison_result(invalid) is None
