# Experiment Summary

This file records the project-level empirical claims that are safe to cite in
the README or a resume. Large raw logs, LoRA weights, generated reports, and
local `outputs/` artifacts are intentionally not committed.

## Head-to-Head: Agent vs Single-Shot LLM

Script: `scripts/run_headtohead.py`

Benchmark: ResearchBench, 15 paired questions from technology, medical, and
finance domains.

Scoring: ResearchBench rule-based composite score using expected topics and
ground-truth facts. The same question is answered by the full Agent pipeline and
by a single direct LLM call.

Result:

| Metric | Value |
|---|---:|
| Agent average | 0.6034 |
| Single-shot baseline average | 0.5586 |
| Absolute delta | +0.0448 |
| Relative lift | about +8.0% |
| 95% bootstrap CI | [+0.0134, +0.0761] |
| p-value | 0.0021 |
| Cohen's d | 0.83 |
| n | 15 |

Raw compact result: `docs/evaluation/headtohead_n15.json`.

## Evaluation Stack

The project uses a two-layer evaluation design:

- Rule metrics for reproducibility and cheap batch runs: semantic/factual
  coverage, hallucination risk, citation coverage, logical consistency,
  comprehensiveness, and efficiency.
- LLM-as-Judge for expert-style audit: factuality, logic, citation quality,
  confidence, and qualitative rationale.

Statistics are reported with paired bootstrap confidence intervals, p-values,
Cohen's d, and paired t-test helpers.

## Citation Quality Validation

The citation-focused validation tracked the main remaining Judge criticism:
"missing complete bibliography and inconsistent citation format." The root
cause was that the synthesis step only saw subtask outputs such as `[Result N]`
instead of structured source metadata.

Fix:

- Extract web and paper sources from subtask trajectories into a shared
  `_collect_sources` helper.
- Preserve arXiv-style paper metadata: title, first authors, publication year,
  PDF/link, snippet, and task id.
- Feed a numbered source list into the synthesis prompt.
- Require body citations to use `[N]` and require a final `## 参考来源` section
  with `[N] title -- author/org (year) -- link`.

Observed Judge trajectory:

| Version | Citation quality | Overall | Notes |
|---|---:|---:|---|
| old baseline | 4/10 | 6/10 | generic or incomplete references |
| v1 | 5/10 | - | researcher prompt prefers academic sources |
| v2 | 6/10 | 7/10 | summarizer cites real sources instead of `[Result N]` |
| v3 | 7/10 | 8/10 | structured source list includes title/author/year/link |

The latest Judge feedback noted that the report cited multiple concrete papers
and conferences. Compact result: `docs/evaluation/citation_quality_v3.json`.

## GRPO Training Study

Script directory: `scripts/grpo_poc/`

Purpose: validate the real training loop behind the M6 self-evolution design.
The PoC uses GSM8K because each rollout has a verifiable reward, making it far
cheaper than training on full multi-step deep-research trajectories.

| Experiment | Model | Setup | Baseline | GRPO after | Conclusion |
|---|---|---|---:|---:|---|
| 1 | Qwen2.5-7B-Instruct | TRL + LoRA, 400 steps | 89% | 91% | pipeline works, gain not significant |
| 2 | Qwen2.5-1.5B-Instruct | TRL + LoRA, 400 steps | 66.6% | 69.4% | +2.8 at n=500, not significant |
| 3 | Qwen2.5-1.5B-Instruct | grad accumulation + temp 1 + 500 steps | 66.6% | 69.4% | reward rose but held-out score did not |
| 4 | Qwen2.5-1.5B base | R1-Zero-style cold start | 55% | 60% | larger headroom gives larger lift, still not p<0.05 |

Safe claim: the GRPO pipeline is real and end-to-end: rollout, verifiable
reward, gradient update, LoRA adapter output, and held-out evaluation all ran.

Unsafe claim: "GRPO significantly improves capability." The available runs did
not reach p<0.05. The most valuable finding is methodological: an early n=200
`+8` result collapsed to `+2.8` at n=500, so the project reports the larger,
less flattering result.

## Engineering Fixes Worth Calling Out

- Unified robust JSON extraction across planner, judge, red agent, blue agent,
  and evolution modules.
- Explicit OpenAI-compatible request timeout and retry bounds.
- Remaining-time timeout wrapper around adversarial review.
- Citation prompt upgrade toward concrete title + author/org + year evidence.
- Targeted Blue Agent edits to avoid whole-report rewrite truncation.
- FileReader sandbox path containment via `Path.is_relative_to`.
- Memory quality filter for greetings, API errors, short junk, and low
  confidence entries.
