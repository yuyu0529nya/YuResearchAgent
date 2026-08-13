from __future__ import annotations

import json

from src.evidence import (
    ClaimRecord,
    ClaimVerifier,
    EvidenceAudit,
    EvidenceKind,
    EvidenceStore,
    VerificationStatus,
    build_evidence_gap_tasks,
)
from src.evidence.store import infer_source_year, source_relevance
from src.orchestrator.schemas import AgentResult, AgentStatus


def _result() -> AgentResult:
    return AgentResult(
        task_id="research_1",
        status=AgentStatus.SUCCESS,
        output="The Transformer architecture relies entirely on attention mechanisms [1].",
        trajectory=[
            {
                "role": "tool",
                "name": "web_search",
                "arguments": {"query": "transformer attention"},
                "result": {
                    "source": "mock",
                    "results": [
                        {
                            "title": "Attention Is All You Need",
                            "url": "https://arxiv.org/abs/1706.03762",
                            "snippet": "The Transformer architecture relies entirely on attention mechanisms.",
                        }
                    ],
                },
            },
            {
                "role": "tool",
                "name": "browser",
                "arguments": {"url": "https://arxiv.org/abs/1706.03762"},
                "result": "The Transformer architecture relies entirely on attention mechanisms. It removes recurrence.",
            },
        ],
        confidence=0.8,
    )


def test_store_ingests_and_deduplicates_search_and_fulltext() -> None:
    store = EvidenceStore(persist_enabled=False)
    store.ingest_results([_result(), _result()])

    assert len(store.sources) == 1
    assert len(store.evidence) == 2
    assert store.source_list()[0].is_primary is True
    assert store.fulltext_source_ids() == {store.source_list()[0].source_id}
    assert {chunk.kind for chunk in store.evidence.values()} == {
        EvidenceKind.SEARCH_SNIPPET,
        EvidenceKind.FULL_TEXT,
    }


def test_store_recognizes_chinese_government_and_university_sources() -> None:
    store = EvidenceStore(persist_enabled=False)

    government = store.upsert_source(url="https://www.gov.cn/zhengce/example")
    ministry = store.upsert_source(url="https://www.moe.gov.cn/srcsite/example")
    university = store.upsert_source(url="https://stem.zjnu.edu.cn/example")
    republisher = store.upsert_source(url="https://www.sohu.com/a/123")

    assert government.is_primary is True
    assert ministry.is_primary is True
    assert university.is_primary is True
    assert republisher.is_primary is False
    assert government.quality_score > republisher.quality_score


def test_government_media_section_is_not_treated_as_primary_document() -> None:
    store = EvidenceStore(persist_enabled=False)

    media = store.upsert_source(
        url="https://jingan.gov.cn/rmtzx/003001/20260715/example.html",
        title="Local integrated-media report",
    )
    policy = store.upsert_source(
        url="https://www.gov.cn/zhengce/202607/content_example.htm",
        title="Official policy document",
    )

    assert media.is_primary is False
    assert media.quality_score == 0.75
    assert policy.is_primary is True
    assert policy.quality_score == 0.95


def test_source_relevance_accepts_one_subtopic_but_rejects_generic_noise() -> None:
    query = "对比中美 STEM 教育政策、课程设计和师资培养，评估双减影响"

    stem_score, stem_hits = source_relevance(
        query,
        "Federal STEM Education Strategic Plan and implementation report",
    )
    noise_score, noise_hits = source_relevance(
        query,
        "China foreign investment and state capitalism in the energy sector",
    )

    assert stem_hits >= 1
    assert stem_score > 0.1
    assert noise_hits == 0
    assert noise_score < stem_score


def test_source_year_uses_html_path_date_before_title() -> None:
    assert infer_source_year(
        "https://www.gov.cn/zhengce/2021/07/24/document.htm",
        "Policy update 2022",
    ) == "2021"


def test_source_year_does_not_use_pdf_upload_directory() -> None:
    assert infer_source_year(
        "https://energy.gov/sites/default/files/2024-03/charting-a-course.pdf",
        "Charting a Course for Success",
    ) == ""


def test_source_year_prefers_explicit_pdf_title_metadata() -> None:
    assert infer_source_year(
        "https://energy.gov/sites/default/files/2024-03/charting-a-course.pdf",
        "Charting a Course for Success (2018)",
    ) == "2018"


def test_store_does_not_treat_repositories_as_primary_sources() -> None:
    store = EvidenceStore(persist_enabled=False)

    github = store.upsert_source(url="https://github.com/example/unverified-repo")
    academia = store.upsert_source(url="https://academia.edu/1234/uploaded-paper")

    assert github.is_primary is False
    assert academia.is_primary is False
    assert github.quality_score < 0.9
    assert academia.quality_score < 0.9


def test_web_host_domain_is_not_exported_as_publisher() -> None:
    store = EvidenceStore(persist_enabled=False)

    source = store.upsert_source(
        url="https://dodstem-assets.dodstem.us/files/federal-plan.pdf",
        title="Federal Strategic Plan",
    )

    assert source.publisher == ""
    assert source.metadata["host_domain"] == "dodstem-assets.dodstem.us"


def test_mismatched_browser_body_is_not_ingested_as_fulltext() -> None:
    result = _result()
    result.trajectory[0]["result"] = {
        "query": "Federal STEM Education Strategic Plan",
        "results": [
            {
                "title": "Federal STEM Education Strategic Plan",
                "url": "https://example.gov/files/plan.pdf",
                "snippet": "Federal STEM strategic plan",
            }
        ],
    }
    result.trajectory[1]["arguments"] = {"url": "https://example.gov/files/plan.pdf"}
    result.trajectory[1]["result"] = (
        "Latest homepage news about wheat cadmium prediction, green ammonia, "
        "urban hydrology, medical imaging, and unrelated foundation announcements."
    )
    store = EvidenceStore(persist_enabled=False)

    store.ingest_results([result])

    assert all(chunk.kind != EvidenceKind.FULL_TEXT for chunk in store.evidence.values())
    assert store.source_list()[0].metadata["fulltext_rejected"] == "content_mismatch"


def test_source_export_marks_fulltext_availability() -> None:
    store = EvidenceStore(persist_enabled=False)
    store.ingest_results([_result()])

    source = store.to_source_dicts()[0]

    assert source["has_fulltext"] is True
    assert source["evidence_count"] == 2
    assert source["evidence_kinds"] == ["full_text", "search_snippet"]


def test_secondary_search_snippet_cannot_establish_a_claim() -> None:
    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(url="https://example.com/summary", title="Secondary summary")
    store.add_evidence(
        source.source_id,
        "The policy reduced tutoring providers by 80 percent.",
        EvidenceKind.SEARCH_SNIPPET,
    )

    audit = ClaimVerifier().audit_text(
        "The policy reduced tutoring providers by 80 percent [1].",
        store,
        citation_source_ids=[source.source_id],
    )

    assert audit.supported_count == 0
    assert audit.claims[0].status == VerificationStatus.NOT_ENOUGH_EVIDENCE
    assert "secondary search snippet" in audit.claims[0].reason


def test_primary_source_snippet_can_establish_an_exact_policy_claim() -> None:
    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(url="https://www.gov.cn/policy/example", title="Policy")
    store.add_evidence(
        source.source_id,
        "The policy reduced tutoring providers by 80 percent.",
        EvidenceKind.SEARCH_SNIPPET,
    )

    audit = ClaimVerifier().audit_text(
        "The policy reduced tutoring providers by 80 percent [1].",
        store,
        citation_source_ids=[source.source_id],
    )

    assert audit.supported_count == 1
    assert audit.claims[0].support_evidence_ids


def test_store_canonicalizes_arxiv_abs_pdf_html_and_versions() -> None:
    store = EvidenceStore(persist_enabled=False)

    first = store.upsert_source(
        url="https://arxiv.org/pdf/2412.15115v2.pdf",
        title="Qwen2.5 Technical Report",
        source_type="web",
    )
    second = store.upsert_source(
        url="https://www.arxiv.org/html/2412.15115",
        title="Qwen2.5 Technical Report",
        source_type="paper",
        authors="Qwen Team",
    )

    assert first.source_id == second.source_id
    assert len(store.sources) == 1
    assert second.url == "https://arxiv.org/abs/2412.15115"
    assert second.source_type == "paper"
    assert second.authors == "Qwen Team"


def test_store_does_not_ingest_browser_error_as_fulltext() -> None:
    result = _result()
    result.trajectory[-1]["result"] = "[Browser Error] HTTP 403 Forbidden"
    store = EvidenceStore(persist_enabled=False)

    store.ingest_results([result])

    assert all(chunk.kind != EvidenceKind.FULL_TEXT for chunk in store.evidence.values())


def test_identical_browser_text_upgrades_search_snippet_to_fulltext() -> None:
    result = _result()
    exact = "The Transformer architecture relies entirely on attention mechanisms."
    result.trajectory[0]["result"]["results"][0]["snippet"] = exact
    result.trajectory[1]["result"] = exact
    store = EvidenceStore(persist_enabled=False)

    store.ingest_results([result])

    assert len(store.evidence) == 1
    assert next(iter(store.evidence.values())).kind == EvidenceKind.FULL_TEXT
    assert store.fulltext_source_ids() == {store.source_list()[0].source_id}


def test_claim_verifier_builds_supported_attribution_edge() -> None:
    store = EvidenceStore(persist_enabled=False)
    result = _result()
    store.ingest_results([result])

    audit = ClaimVerifier().audit_results([result], store)

    assert audit.supported_count == 1
    assert audit.coverage == 1.0
    claim = audit.claims[0]
    assert claim.status == VerificationStatus.SUPPORTED
    assert claim.support_evidence_ids
    assert claim.source_ids == [store.source_list()[0].source_id]


def test_fast_presynthesis_audit_skips_hybrid_policy() -> None:
    class _Policy:
        def __call__(self, _):
            raise AssertionError("LLM should not run during fast audit")

    store = EvidenceStore(persist_enabled=False)
    result = _result()
    store.ingest_results([result])

    audit = ClaimVerifier(policy=_Policy(), mode="hybrid").audit_results(
        [result], store, use_llm=False
    )

    assert audit.verification_mode == "heuristic"
    assert audit.supported_count == 1


def test_citation_restricts_verification_to_attributed_source() -> None:
    store = EvidenceStore(persist_enabled=False)
    first = store.upsert_source(url="https://example.com/unrelated", title="Unrelated")
    store.add_evidence(first.source_id, "A completely unrelated article about cooking.", EvidenceKind.FULL_TEXT)
    second = store.upsert_source(url="https://arxiv.org/abs/1706.03762", title="Attention")
    store.add_evidence(
        second.source_id,
        "The Transformer architecture relies entirely on attention mechanisms.",
        EvidenceKind.ABSTRACT,
    )

    audit = ClaimVerifier().audit_text(
        "The Transformer architecture relies entirely on attention mechanisms [1].",
        store,
        citation_source_ids=[first.source_id, second.source_id],
    )

    assert audit.claims[0].status == VerificationStatus.NOT_ENOUGH_EVIDENCE


def test_reference_list_is_not_extracted_as_claims() -> None:
    text = (
        "A sufficiently long factual statement appears in the report [1].\n\n"
        "## References\n"
        "[1] A very long bibliography entry that should not become a claim."
    )

    claims = ClaimVerifier().extract_claims(text)

    assert len(claims) == 1
    assert "bibliography" not in claims[0].text


def test_process_narration_is_not_counted_as_external_claim() -> None:
    text = (
        "本报告仅依据 Qwen2.5 官方技术报告进行核验。\n"
        "两份子任务核验结果在该数字上完全一致。\n"
        "Qwen2.5 was pretrained on 18 trillion tokens [1]."
    )

    claims = ClaimVerifier().extract_claims(text)

    assert [claim.text for claim in claims] == [
        "Qwen2.5 was pretrained on 18 trillion tokens ."
    ]


def test_discourse_prefix_and_bold_enumerator_do_not_hide_fact() -> None:
    claims = ClaimVerifier().extract_claims(
        "结论如下：**(1) 预训练规模已确认**：Qwen2.5 使用了 18 万亿 token [1]；"
    )

    assert len(claims) == 1
    assert claims[0].cited_indices == [1]
    assert not claims[0].text.startswith("结论如下")
    assert "18 万亿" in claims[0].text


def test_chinese_claims_split_without_spaces_after_punctuation() -> None:
    text = "Transformer 完全依赖注意力机制进行序列建模。该架构移除了循环神经网络结构。"

    claims = ClaimVerifier().extract_claims(text)

    assert len(claims) == 2


def test_claims_split_at_semicolon_into_atomic_units() -> None:
    text = (
        "Qwen2.5 was trained on 18 trillion tokens; "
        "Gemini 1.5 supports a million-token context window."
    )

    claims = ClaimVerifier().extract_claims(text)

    assert len(claims) == 2


def test_gap_planner_targets_unresolved_claims() -> None:
    store = EvidenceStore(persist_enabled=False)
    audit = ClaimVerifier().audit_text(
        "The benchmark improved accuracy by 8 percent across fifteen evaluated tasks.",
        store,
    )

    tasks = build_evidence_gap_tasks(audit, round_index=1, max_tasks=2)

    assert len(tasks) == 1
    assert tasks[0].task_id == "evidence_gap_r1_1"
    assert "8 percent" in tasks[0].description
    assert tasks[0].expected_type == "verification"


def test_gap_planner_prefers_query_central_claim_over_numeric_side_claim() -> None:
    audit = EvidenceAudit(
        claims=[
            ClaimRecord(
                claim_id="side",
                text="Figure AI completed a 675 million dollar funding round in 2025.",
            ),
            ClaimRecord(
                claim_id="central",
                text="具身智能的关键技术包括世界模型、触觉感知与机器人任务规划。",
            ),
        ],
        nei_count=2,
    )

    tasks = build_evidence_gap_tasks(
        audit,
        round_index=1,
        max_tasks=1,
        query="分析具身智能中的世界模型、触觉感知与机器人任务规划",
    )

    assert len(tasks) == 1
    assert "世界模型" in tasks[0].description
    assert "675" not in tasks[0].description


def test_store_persists_auditable_json(tmp_path) -> None:
    store = EvidenceStore(artifact_dir=str(tmp_path), session_id="test", persist_enabled=True)
    result = _result()
    store.ingest_results([result])
    audit = ClaimVerifier().audit_results([result], store)

    path = store.persist(audit, query="How does Transformer work?")
    payload = json.loads(open(path, encoding="utf-8").read())

    assert payload["schema_version"] == "1.0"
    assert payload["audit"]["coverage"] == 1.0
    assert payload["sources"][0]["source_id"].startswith("src_")
    assert payload["evidence"][0]["evidence_id"].startswith("ev_")


def test_store_loads_persisted_graph_without_changing_ids(tmp_path) -> None:
    store = EvidenceStore(artifact_dir=str(tmp_path), session_id="roundtrip", persist_enabled=True)
    store.ingest_results([_result()])
    path = store.persist(query="How does Transformer work?")

    loaded = EvidenceStore.load_artifact(path)

    assert loaded.query == "How does Transformer work?"
    assert list(loaded.sources) == list(store.sources)
    assert list(loaded.evidence) == list(store.evidence)
    assert loaded.persist_enabled is False


def test_store_rejects_tampered_evidence_artifact(tmp_path) -> None:
    store = EvidenceStore(artifact_dir=str(tmp_path), session_id="tamper", persist_enabled=True)
    store.ingest_results([_result()])
    path = store.persist()
    payload = json.loads(open(path, encoding="utf-8").read())
    payload["evidence"][0]["text"] = "tampered"
    open(path, "w", encoding="utf-8").write(json.dumps(payload))

    try:
        EvidenceStore.load_artifact(path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered evidence should fail integrity verification")


def test_cited_evidence_audit_narration_is_not_a_claim() -> None:
    claims = ClaimVerifier().extract_claims(
        "该两项方法均在证据审计中获得直接支持 [1]。\n"
        "Qwen2.5 was pretrained on 18 trillion tokens [1]."
    )

    assert [claim.text for claim in claims] == [
        "Qwen2.5 was pretrained on 18 trillion tokens ."
    ]


def test_bibliographic_fragments_are_not_extracted_as_claims() -> None:
    claims = ClaimVerifier().extract_claims(
        "Kawaharazuka et al., IEEE Access, 2025）。\n"
        "Yang Li《World Model：从强化学习到具身智能的完整入门》教程）。\n"
        "世界模型通过预测未来状态辅助机器人规划。"
    )

    assert [claim.text for claim in claims] == [
        "世界模型通过预测未来状态辅助机器人规划。"
    ]


def test_official_documentation_is_classified_as_primary() -> None:
    store = EvidenceStore(persist_enabled=False)

    source = store.upsert_source(
        url="https://platform.example.com/docs/guide/model",
        title="Official model guide",
    )

    assert source.is_primary is True
    assert source.quality_score == 0.85


def test_numeric_verifier_tolerates_extraction_space_after_decimal() -> None:
    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(
        url="https://platform.example.com/docs/model",
        title="Official model documentation",
        source_type="web",
        task_id="t1",
    )
    store.add_evidence(
        source.source_id,
        "模型有 2. 8 万亿参数和 1, 000 token 上下文。",
        EvidenceKind.FULL_TEXT,
        "t1",
    )

    audit = ClaimVerifier(support_threshold=0.2).audit_text(
        "模型有 2.8万亿参数和 1,000 token 上下文 [1]。",
        store,
        citation_source_ids=[source.source_id],
    )

    assert audit.supported_count == 1


def test_numeric_verifier_normalizes_chinese_and_english_scale_units() -> None:
    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(url="https://arxiv.org/abs/test", title="Model report")
    store.add_evidence(
        source.source_id,
        "The model was trained on 18 trillion tokens.",
        EvidenceKind.ABSTRACT,
    )

    audit = ClaimVerifier(support_threshold=0.1).audit_text(
        "该模型使用 18 万亿 token 进行训练 [1]。",
        store,
        citation_source_ids=[source.source_id],
    )

    assert "numeric details do not match" not in audit.claims[0].reason


def test_hybrid_verifier_downgrades_partial_compound_support() -> None:
    class _OverconfidentPolicy:
        tools = None

        def __call__(self, messages):
            prompt = messages[-1]["content"]
            payload = json.loads(prompt.rsplit("\n\n", 1)[-1])
            return {
                "content": json.dumps(
                    {
                        "verdicts": [
                            {
                                "claim_id": payload[0]["claim_id"],
                                "label": "SUPPORTED",
                                "score": 0.95,
                                "reason": "Looks supported",
                            }
                        ]
                    }
                )
            }

    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(url="https://example.com/model", title="Model card")
    store.add_evidence(
        source.source_id,
        "The model uses a mixture-of-experts architecture.",
        EvidenceKind.FULL_TEXT,
    )
    audit = ClaimVerifier(
        policy=_OverconfidentPolicy(),
        mode="hybrid",
        support_threshold=0.95,
    ).audit_text(
        "The model uses a mixture-of-experts architecture and was trained on 18 trillion tokens.",
        store,
    )

    assert audit.claims[0].status == VerificationStatus.NOT_ENOUGH_EVIDENCE
    assert "downgraded" in audit.claims[0].reason


def test_hybrid_verifier_can_check_cited_cross_language_evidence() -> None:
    class _BilingualPolicy:
        tools = None

        def __call__(self, messages):
            payload = json.loads(messages[-1]["content"].rsplit("\n\n", 1)[-1])
            return {
                "content": json.dumps(
                    {
                        "verdicts": [
                            {
                                "claim_id": payload[0]["claim_id"],
                                "label": "SUPPORTED",
                                "clause_labels": ["SUPPORTED"],
                                "score": 0.92,
                                "reason": "The English evidence entails the Chinese claim.",
                            }
                        ]
                    }
                )
            }

    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(url="https://arxiv.org/abs/2412.15115", title="Qwen2.5")
    store.add_evidence(
        source.source_id,
        "Qwen2.5 was pretrained on up to 18 trillion tokens.",
        EvidenceKind.ABSTRACT,
    )
    audit = ClaimVerifier(
        policy=_BilingualPolicy(),
        mode="hybrid",
        support_threshold=0.95,
    ).audit_text(
        "Qwen2.5 使用了 18 万亿 token 进行预训练 [1]。",
        store,
        citation_source_ids=[source.source_id],
    )

    assert audit.claims[0].candidate_evidence_ids
    assert audit.claims[0].status == VerificationStatus.SUPPORTED


def test_same_task_provenance_recalls_cross_language_abstract_for_hybrid_audit() -> None:
    class _BilingualPolicy:
        tools = None

        def __call__(self, messages):
            payload = json.loads(messages[-1]["content"].rsplit("\n\n", 1)[-1])
            return {
                "content": json.dumps(
                    {
                        "verdicts": [
                            {
                                "claim_id": payload[0]["claim_id"],
                                "label": "SUPPORTED",
                                "clause_labels": ["SUPPORTED"],
                                "score": 0.91,
                                "reason": "The abstract entails the translated claim.",
                            }
                        ]
                    }
                )
            }

    result = AgentResult(
        task_id="embodied_ai",
        status=AgentStatus.SUCCESS,
        output="触觉反馈能够提升机器人操作策略对接触状态的感知能力。",
        trajectory=[
            {
                "role": "tool",
                "name": "arxiv_reader",
                "arguments": {"query": "tactile sensing robot manipulation"},
                "result": {
                    "source": "fixture",
                    "query": "tactile sensing robot manipulation",
                    "papers": [
                        {
                            "title": "Tactile sensing for robot manipulation",
                            "url": "https://arxiv.org/abs/2501.00001",
                            "summary": (
                                "Tactile feedback improves a manipulation policy's perception "
                                "of physical contact states."
                            ),
                            "_relevance_score": 0.9,
                        }
                    ],
                },
            }
        ],
    )
    store = EvidenceStore(persist_enabled=False)
    store.ingest_results([result])

    heuristic = ClaimVerifier(mode="hybrid", support_threshold=0.95).audit_results(
        [result], store, use_llm=False
    )
    hybrid = ClaimVerifier(
        policy=_BilingualPolicy(), mode="hybrid", support_threshold=0.95
    ).audit_results([result], store, use_llm=True)

    assert heuristic.claims[0].candidate_evidence_ids
    assert heuristic.claims[0].status == VerificationStatus.NOT_ENOUGH_EVIDENCE
    assert hybrid.claims[0].status == VerificationStatus.SUPPORTED


def test_hybrid_verifier_marks_missing_verdicts_unavailable() -> None:
    class _EmptyPolicy:
        tools = None

        def __call__(self, _messages):
            return {"content": '{"verdicts": []}'}

    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(
        url="https://arxiv.org/abs/2501.00001",
        title="Tactile sensing for robot manipulation",
        source_type="paper",
    )
    store.add_evidence(
        source.source_id,
        "Tactile feedback improves perception of physical contact states.",
        EvidenceKind.ABSTRACT,
    )

    audit = ClaimVerifier(
        policy=_EmptyPolicy(),
        mode="hybrid",
        support_threshold=0.95,
    ).audit_text(
        "触觉反馈能够提升机器人对接触状态的感知能力 [1]。",
        store,
        citation_source_ids=[source.source_id],
    )

    assert audit.verification_mode == "hybrid_unavailable"
    assert audit.claims[0].status == VerificationStatus.NOT_ENOUGH_EVIDENCE


def test_hybrid_verifier_marks_sampled_candidates_partial() -> None:
    class _OneVerdictPolicy:
        tools = None

        def __call__(self, messages):
            payload = json.loads(messages[-1]["content"].rsplit("\n\n", 1)[-1])
            return {
                "content": json.dumps(
                    {
                        "verdicts": [
                            {
                                "claim_id": payload[0]["claim_id"],
                                "label": "NOT_ENOUGH_EVIDENCE",
                                "clause_labels": ["NOT_ENOUGH_EVIDENCE"],
                                "score": 0.2,
                                "reason": "Insufficient detail.",
                            }
                        ]
                    }
                )
            }

    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(
        url="https://arxiv.org/abs/2501.00003",
        title="Robotics evidence",
        source_type="paper",
    )
    store.add_evidence(
        source.source_id,
        "Robotics evidence discusses tactile feedback and long-horizon planning.",
        EvidenceKind.ABSTRACT,
    )
    verifier = ClaimVerifier(
        policy=_OneVerdictPolicy(),
        mode="hybrid",
        support_threshold=0.99,
        max_llm_claims=1,
    )

    audit = verifier.audit_text(
        (
            "Tactile feedback improves contact perception [1]. "
            "Long-horizon planning remains unreliable [1]."
        ),
        store,
        citation_source_ids=[source.source_id],
    )

    assert audit.verification_mode == "hybrid_partial"
    assert audit.semantic_reviewed_count == 1
    assert audit.semantic_candidate_count == 2


def test_temporal_labels_and_short_heading_fragments_are_not_claims() -> None:
    claims = ClaimVerifier().extract_claims(
        "（信息截至 2026-08-14）\n"
        "**Sim-to-real 鸿沟。**\n"
        "真实机器人试错可能耗时且带来安全风险 [1]。"
    )

    assert [claim.text for claim in claims] == [
        "真实机器人试错可能耗时且带来安全风险 。"
    ]


def test_lexical_negation_does_not_deterministically_refute_claim() -> None:
    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(
        url="https://arxiv.org/abs/2501.00002",
        title="Tactile data processing",
        source_type="paper",
    )
    store.add_evidence(
        source.source_id,
        "Efficient tactile data processing is not yet a solved problem.",
        EvidenceKind.ABSTRACT,
    )

    audit = ClaimVerifier(support_threshold=0.2).audit_text(
        "Efficient tactile data processing is a solved problem [1].",
        store,
        citation_source_ids=[source.source_id],
    )

    assert audit.refuted_count == 0
    assert audit.claims[0].status == VerificationStatus.NOT_ENOUGH_EVIDENCE
    assert "semantic verification" in audit.claims[0].reason


def test_task_provenance_does_not_cross_agent_boundaries() -> None:
    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(
        url="https://arxiv.org/abs/2501.00001",
        title="Tactile sensing for robot manipulation",
        source_type="paper",
        task_id="other_task",
    )
    store.add_evidence(
        source.source_id,
        "Tactile feedback improves perception of physical contact states.",
        EvidenceKind.ABSTRACT,
        task_id="other_task",
    )
    result = AgentResult(
        task_id="embodied_ai",
        status=AgentStatus.SUCCESS,
        output="触觉反馈能够提升机器人对接触状态的感知能力。",
    )

    audit = ClaimVerifier(mode="hybrid", support_threshold=0.95).audit_results(
        [result], store, use_llm=False
    )

    assert audit.claims[0].candidate_evidence_ids == []
    assert audit.claims[0].status == VerificationStatus.NOT_ENOUGH_EVIDENCE


def test_hybrid_verifier_forwards_request_deadline() -> None:
    class _DeadlinePolicy:
        def __init__(self) -> None:
            self.tools = ["tool"]
            self.timeout = None

        def __call__(self, _messages):
            raise AssertionError("bounded audits must use call_with_timeout")

        def call_with_timeout(self, messages, timeout_seconds):
            assert self.tools is None
            self.timeout = timeout_seconds
            payload = json.loads(messages[-1]["content"].rsplit("\n\n", 1)[-1])
            return {
                "content": json.dumps(
                    {
                        "verdicts": [
                            {
                                "claim_id": payload[0]["claim_id"],
                                "label": "SUPPORTED",
                                "clause_labels": ["SUPPORTED"],
                                "score": 0.92,
                                "reason": "The evidence entails the claim.",
                            }
                        ]
                    }
                )
            }

    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(
        url="https://arxiv.org/abs/2412.15115",
        title="Qwen2.5 Technical Report",
    )
    store.add_evidence(
        source.source_id,
        "Qwen2.5 was pretrained on up to 18 trillion tokens.",
        EvidenceKind.ABSTRACT,
    )
    policy = _DeadlinePolicy()

    audit = ClaimVerifier(
        policy=policy,
        mode="hybrid",
        support_threshold=0.95,
    ).audit_text(
        "Qwen2.5 使用了 18 万亿 token 进行预训练 [1]。",
        store,
        citation_source_ids=[source.source_id],
        timeout_seconds=2.0,
    )

    assert audit.claims[0].status == VerificationStatus.SUPPORTED
    assert policy.timeout is not None
    assert 0.25 <= policy.timeout <= 2.0
    assert policy.tools == ["tool"]


def test_candidate_ranking_prioritizes_matching_numeric_chunk() -> None:
    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(url="https://arxiv.org/abs/2412.15115", title="Qwen2.5")
    unrelated = store.add_evidence(
        source.source_id,
        "Qwen2.5 is a family of language models with several parameter sizes.",
        EvidenceKind.FULL_TEXT,
    )
    matching = store.add_evidence(
        source.source_id,
        "The pre-training dataset was expanded to 18 trillion tokens.",
        EvidenceKind.FULL_TEXT,
    )

    audit = ClaimVerifier(support_threshold=0.95).audit_text(
        "Qwen2.5 使用 18 万亿 token 进行预训练 [1]。",
        store,
        citation_source_ids=[source.source_id],
        use_llm=False,
    )

    assert audit.claims[0].candidate_evidence_ids[0] == matching.evidence_id
    assert audit.claims[0].candidate_evidence_ids[0] != unrelated.evidence_id
