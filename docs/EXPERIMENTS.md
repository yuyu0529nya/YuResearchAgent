# Experiment and Claim Ledger

This ledger records which project claims are reproducible, which are compact
development traces, and which have been superseded after code audit.

## Evidence Levels

- **Reproducible**: code, configuration, and a committed compact artifact exist.
- **Locally auditable**: large raw artifacts exist locally and are identified by
  SHA-256, while the compact manifest is committed.
- **Development trace**: useful for debugging, but not a population-level result.
- **Superseded**: a protocol defect invalidates the original comparison claim.

## ResearchBench v2 Metric Contract

The rule composite intends to assign a 25% weight to factual accuracy. Before
2026-08-13, `ResearchBench.evaluate_report()` emitted
`factual_accuracy_str` and `factual_accuracy_sem`, while
`RuleBasedMetrics.composite_score()` looked for `factual_accuracy`. The lookup
therefore returned zero and silently removed the entire factuality component.

The fix is now covered by tests:

```text
factual_accuracy = 0.4 * factual_accuracy_str
                 + 0.6 * factual_accuracy_sem
```

New head-to-head artifacts identify themselves as
`researchbench-v2-factual-weight-fix` and include metric weights.

## Superseded Head-to-Head Result

The archived n=15 artifact contains these historical values:

| Metric | Legacy value |
|---|---:|
| Agent average | 0.6034 |
| Single-shot average | 0.5586 |
| Difference | +0.0448 |
| Bootstrap 95% CI | [+0.0134, +0.0761] |
| p-value | 0.0021 |
| Cohen's d | 0.83 |

Status: **superseded pending rerun**. The compact artifact does not retain the
paired reports, so the corrected factual score cannot be reconstructed. These
numbers may be cited only as a historical debugging result, not as evidence that
the multi-agent system outperforms a single-shot model.

Artifact: `docs/evaluation/headtohead_n15.json`.

## Corrected Head-to-Head Development Pilot

One `tech_001` pair has completed under the corrected v3 protocol. Both systems
used Kimi K3; the Agent used the full tool/evidence runtime, while the baseline
received one model call and no tools. DeepSeek judged the retained reports in
both A/B presentation orders.

| n=1 measure | Agent | Baseline | Delta |
|---|---:|---:|---:|
| ResearchBench v2 rule composite | 0.7182 | 0.6430 | +0.0753 |
| Counterbalanced Judge mean (1-5) | 4.50 | 3.25 | +1.25 |
| Elapsed seconds | 447.70 | 52.04 | +395.66 |
| API tokens | 54,305 | 1,739 | +52,566 |

Status: **development pilot, not evidence of general superiority**. At `n=1`,
the statistics layer emits `p=1.0` and `method=insufficient_n`. The exact reports,
evidence graph, raw Judge decisions, token counts, and SHA-256 values are retained
under `docs/evaluation/artifacts/headtohead_v3/`.

## Real Kimi K3 Evidence Run And Replay

Date: 2026-08-13. Query: verify Qwen2.5 pre-training scale, context windows,
and post-training methods using first-party technical reports.

| Runtime metric | Result |
|---|---:|
| Run status | complete |
| API calls / tokens | 11 / 41,267 |
| Search calls | 3 |
| Sources / evidence chunks | 3 / 15 |
| Primary-source ratio | 100% |
| Full-text-source ratio | 33.3% |
| Elapsed | 352.76 seconds |

The exact report and evidence graph are committed. Under the current claim
extractor, deterministic replay supports 3/30 claims (10.0%), while a batched,
source-bounded DeepSeek replay supports 13/30 (43.3%). The difference reflects
cross-language semantic verification; both modes retain unsupported compound
claims as NEI.

Status: **reproducible runtime/verifier evidence, not a quality benchmark**.
Artifacts: `docs/evaluation/artifacts/evidence_v2/`.

## Module Ablation Diagnostic

A real `full` versus `no_evidence` pair completed on `tech_001`:

| n=1 measure | Full | No evidence | Full minus no evidence |
|---|---:|---:|---:|
| Rule composite | 0.7289 | 0.7405 | -0.0116 |
| Elapsed seconds | 519.08 | 415.32 | +103.76 |
| API tokens | 67,056 | 42,029 | +25,027 |

Status: **negative development diagnostic**. It does not establish an evidence
module quality gain and predates the latest full-text and verifier fixes. Exact
reports and evidence are retained under `docs/evaluation/artifacts/ablation_v3/`.

## Citation Quality Development Trace

A fixed Judge run tracked one report through three prompt/source-provenance
iterations:

| Version | Citation quality | Overall | Change |
|---|---:|---:|---|
| Baseline | 4/10 | 6/10 | generic or incomplete references |
| v1 | 5/10 | not recorded | prefer academic sources |
| v2 | 6/10 | 7/10 | cite real sources rather than result labels |
| v3 | 7/10 | 8/10 | title/author/year/URL catalog in synthesis |

Status: **development trace**. It supports the causal debugging narrative for
one validation case, not a general +3 point citation-quality claim.

Artifact: `docs/evaluation/citation_quality_v3.json`.

## GRPO Training Study

Four TRL + LoRA + GRPO runs were completed on a single RTX 5090 using GSM8K
verifiable correctness and format rewards. Large logs and adapter weights are
intentionally ignored by Git; the committed manifest records results and hashes
for the local originals.

| Run | Model | Steps | Evaluation | Baseline | Post | Interpretation |
|---|---|---:|---:|---:|---:|---|
| 1 | Qwen2.5-7B-Instruct | 300 | n=100 | 89% | 91% | aggregate gain only; paired significance cannot be reconstructed |
| 2 | Qwen2.5-1.5B-Instruct | 400 | n=500 | 66.6% | 69.4% | +2.8 points, approximate independent `p≈0.34` |
| 3 | Qwen2.5-1.5B-Instruct | 500 | n=500 | 66.6% | 69.4% | higher train reward, unchanged held-out accuracy |
| 4 | Qwen2.5-1.5B base | 500 | n=300 | 55% | 60% | +5 points, approximate independent `p≈0.21` |

Status: **locally auditable**. Adapter configs, adapter weights, training logs,
and evaluation logs exist locally and match the hashes in
`docs/evaluation/grpo_runs.json`.

Defensible claims:

- rollout, verifiable reward, gradient update, LoRA export, and held-out
  evaluation completed for all four configurations;
- the n=200 +8 point result for 1.5B shrank to +2.8 at n=500;
- raising training reward in run 3 did not improve its held-out result over run 2;
- none of the retained comparisons establishes a significant capability gain.

The GRPO PoC validates a training pipeline. It is not currently connected to
online policy updates in the research-agent runtime.

## Next Required Experiments

1. Rerun all 15 paired head-to-head questions under ResearchBench v2 and retain
   both generated reports plus per-metric rows.
2. Compare `evidence.enabled`, evidence-gap rounds, and verification modes on a
   fixed set with source/citation precision, claim coverage, latency, and cost.
3. Ablate evidence-bounded revision on that fixed set, retaining every original,
   candidate, gate decision, final report, and API-cost delta.
4. Add a small expert-authored binary-rubric subset modeled after DeepResearch
   Bench II, without claiming compatibility with that benchmark.
5. Preserve per-question GRPO predictions so paired tests can be computed rather
   than inferred from aggregate accuracies.
