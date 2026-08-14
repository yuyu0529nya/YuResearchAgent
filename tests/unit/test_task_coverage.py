from src.evidence.task_coverage import audit_task_coverage


def test_task_coverage_distinguishes_synthesis_and_research_gaps() -> None:
    audit = audit_task_coverage(
        """## Accuracy\nRetrieval accuracy improves when the system reranks evidence.\n""",
        [
            {
                "task_id": "task_1",
                "description": "Compare retrieval accuracy and evidence reranking.",
                "status": "success",
            },
            {
                "task_id": "task_2",
                "description": "Assess operating latency and cost trade-offs.",
                "status": "success",
            },
            {
                "task_id": "task_3",
                "description": "Review governance and compliance risks.",
                "status": "failed",
            },
        ],
        [
            {"task_ids": ["task_1"]},
            {"task_ids": ["task_2"]},
        ],
    )

    assert audit["required_count"] == 3
    assert audit["covered_count"] == 1
    assert audit["coverage"] == 1 / 3
    assert audit["synthesis_gap_count"] == 1
    assert audit["research_gap_count"] == 1
    assert [item["outcome"] for item in audit["items"]] == [
        "covered",
        "synthesis_gap",
        "research_gap",
    ]


def test_task_coverage_ignores_bibliography_terms() -> None:
    audit = audit_task_coverage(
        """## Finding\nThe report discusses deployment choices.\n\n## References\n[1] Governance compliance framework.""",
        [
            {
                "task_id": "task_1",
                "description": "Review governance compliance framework requirements.",
                "status": "success",
            },
        ],
        [{"task_ids": ["task_1"]}],
    )

    assert audit["coverage"] == 0.0
    assert audit["items"][0]["outcome"] == "synthesis_gap"
