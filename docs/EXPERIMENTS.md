# Experiment and Claim Ledger

This ledger records which project claims are reproducible, which are compact
development traces, and which have been superseded after code audit.

## Evidence Levels

- **Reproducible**: code, configuration, and committed auditable artifacts exist.
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

Status: **superseded**. The compact artifact does not retain the
paired reports, so the corrected factual score cannot be reconstructed. These
numbers may be cited only as a historical debugging result, not as evidence that
the multi-agent system outperforms a single-shot model. The corrected v5 result
below replaces this claim.

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

## Preregistered Confirmatory Result

The v5 manifest was committed before execution at source commit `23cfb11`. It
freezes 15 unseen questions spanning all 11 ResearchBench domains, alternating
Agent/baseline generation order, both Judge presentation orders, Kimi K3
effective sampling, DeepSeek Judge identity, the as-of date, prompts, Agent
configuration, evaluator implementation, and source-tree hashes. All 15 pairs
completed; no question was replaced and no significance-based early stopping
was used.

| Confirmatory measure (`n=15`) | Agent | Baseline | Difference |
|---|---:|---:|---:|
| ResearchBench v2 rule composite | 0.6803 | 0.6174 | +0.0629 |
| Counterbalanced Judge mean (1-5) | 4.2000 | 4.2417 | -0.0417 |
| Factual accuracy | 0.3556 | 0.3378 | +0.0178 |
| Citation coverage | 0.3700 | 0.0000 | +0.3700 |
| Comprehensiveness rule metric | 0.8742 | 0.9266 | -0.0524 |
| Mean elapsed seconds | 246.94 | 75.26 | +171.68 |
| Mean API tokens | 75,301 | 2,850 | +72,451 |

Primary endpoint: 95% bootstrap CI `[+0.0452, +0.0828]`, exact one-sided
paired sign-flip `p=0.000061`, paired Cohen's `d_z=1.6477`, 14 wins and 1 loss.
This confirms a rule-composite gain under the frozen metric contract.

Secondary endpoint: Judge CI `[-0.4417, +0.3667]`, exact one-sided sign-flip
`p=0.593018`, paired `d_z=-0.0503`, 5 wins, 3 ties, and 7 losses. Mean Judge
source-quality favored the Agent (`3.80` vs `3.07`), while completeness
(`4.40` vs `4.73`), accuracy (`4.17` vs `4.47`), and structure (`4.43` vs
`4.70`) favored the baseline. The run therefore does not establish a general
answer-quality improvement.

Strict evidence metrics expose the remaining bottleneck: claim-support coverage
`5.79%`, claim citation rate `30.28%`, cited-claim support precision `16.67%`,
primary-source ratio `11.12%`, and full-text-source ratio `7.19%`. The one-call
baseline had no formal references, which strongly drives the rule-composite
gain. The Agent also used 1,129,514 tokens versus 42,749 for baseline generation
and averaged 3.28x the wall time.

Status: **reproducible, complete, independently audited**. The API-free auditor
returned `valid=true`, `complete=true`, 15/15 pairs, no errors, and no warnings.

Artifacts:

- preregistration: `docs/evaluation/headtohead_v5_preregistration.json`;
- portable audited result: `docs/evaluation/artifacts/headtohead_v5/result.json`;
- raw runner result: `docs/evaluation/artifacts/headtohead_v5/result.raw.json`;
- exact Agent/baseline reports and evidence graphs:
  `docs/evaluation/artifacts/headtohead_v5/reports/`.

## v6 Development Coverage Diagnostic

After v5, a task-coverage contract and deterministic audit were added to separate
missing research from synthesis omissions. A frozen three-question development
run (`edu_001`, `cross_002`, `tech_005`) produced a rule-composite difference of
`+0.0502`, but the counterbalanced Judge favored the baseline by `0.2083/5`.
With `n=3`, the exact paired sign-flip result is `p=0.125`; this is diagnostic
data, not a significance claim.

The task audit found `0` synthesis gaps. Its failures were all research gaps:
one of four planned dimensions had no usable task result on the education case,
and two of four were unavailable on each technology/AI-drug case. Mean strict
claim-support coverage was only `5.52%`. The conclusion is therefore negative
but useful: the coverage prompt carried retrieved dimensions into the report,
yet per-task timeouts and weak source acquisition prevent a Judge-quality lift.

The frozen manifest, complete result, reports, evidence artifacts, and raw Judge
decisions are retained in `docs/evaluation/artifacts/headtohead_v6_dev/`.

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

1. Compare `evidence.enabled`, evidence-gap rounds, and verification modes on a
   fixed set with source/citation precision, claim coverage, latency, and cost.
2. Ablate evidence-bounded revision on that fixed set, retaining every original,
   candidate, gate decision, final report, and API-cost delta.
3. Add a small expert-authored binary-rubric subset modeled after DeepResearch
   Bench II, without claiming compatibility with that benchmark.
4. Improve primary/full-text source yield and rerun a frozen citation-support
   benchmark; the current n=15 strict support coverage is only 5.79%.
5. Preserve per-question GRPO predictions so paired tests can be computed rather
   than inferred from aggregate accuracies.
