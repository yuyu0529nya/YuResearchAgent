# Evaluation Artifact Index

This directory separates reproducible evidence from development traces and
superseded results. See `../EXPERIMENTS.md` for interpretation and limitations.

| Artifact | Evidence level | What it establishes |
|---|---|---|
| `demos/catalog.json` | Verified public demo | SHA-256-bound Kimi K3 report/evidence replay available in the UI without an API key |
| `headtohead_v5_preregistration.json` | Preregistered protocol (completed) | Frozen n=15 sample, order, model/config identity, endpoints, analysis plan, as-of date, and source hash across all 11 domains |
| `artifacts/headtohead_v5/` | Reproducible confirmatory result (`n=15`) | Exact reports/evidence, raw Judge decisions, hashes, telemetry, and a portable result that passes the API-free auditor |
| `artifacts/headtohead_v6_dev/` | Development coverage diagnostic (`n=3`) | Frozen three-question diagnostic for task coverage; rule gain retained, Judge regression and worker timeout bottleneck documented |
| `headtohead_v6_concurrency_dev_preregistration.json` | Frozen development protocol | Same three questions and scoring protocol as v6; isolates four-worker concurrent startup as the only Agent intervention |
| `headtohead_v7_resilience_dev_preregistration.json` | Frozen development protocol | Same three questions and scoring protocol; evaluates four-worker startup with provider-level DDGS throttling after the v6 infrastructure failure |
| `headtohead_v8_resilience_retry_dev_preregistration.json` | Frozen development protocol | Same three questions and scoring protocol; evaluates the concurrency, search-throttling, and bounded-retry resilience stack |
| `headtohead_v9_parallel_resilience_dev_preregistration.json` | Frozen development protocol | Same three questions and scoring protocol; adds the independent-search DAG constraint to the resilience stack |
| `headtohead_v10_openrouter_search_dev_preregistration.json` | Frozen development protocol | Same three questions and scoring protocol; records the effective OpenRouter server-side search backend and proxy-aware runtime |
| `artifacts/evidence_v2/` | Reproducible runtime/verifier evidence | Exact Kimi K3 report and evidence graph, deterministic replay, and source-bounded hybrid replay |
| `artifacts/headtohead_v3/` | Development pilot (`n=1`) | Corrected Agent-versus-one-call protocol, retained reports, hashes, telemetry, and counterbalanced Judge output |
| `artifacts/ablation_v3/` | Negative development diagnostic (`n=1`) | Real `full` versus `no_evidence` execution; no quality gain established |
| `citation_quality_v3.json` | Development trace | One fixed citation-prompt iteration from 4/10 to 7/10 |
| `grpo_runs.json` | Locally auditable manifest | Four local GRPO runs and hashes of ignored training artifacts |
| `headtohead_n15.json` | Superseded | Historical result invalidated by a factuality-weight contract bug |

## Replay

The current verifier can replay the committed evidence pair without network or
model access:

```bash
python scripts/replay_evidence.py \
  --report docs/evaluation/artifacts/evidence_v2/qwen25_agent.md \
  --evidence docs/evaluation/artifacts/evidence_v2/qwen25_evidence.json \
  --output outputs/evaluation/evidence_replay.json
```

Add `--verifier-backend <configured-backend>` for strict semantic verification.
The CLI verifies optional expected input hashes before auditing and validates
every evidence chunk against the hash stored inside the graph.

## Preregistered Head-To-Head

The v5 manifest was committed before the confirmatory run. The runner does not
accept question-selection or Judge-order overrides. The published run completed
all 15 pairs and is auditable directly from a clone:

```bash
python scripts/audit_headtohead.py \
  docs/evaluation/artifacts/headtohead_v5/result.json \
  --preregistration docs/evaluation/headtohead_v5_preregistration.json \
  --require-complete
```

The command is API-free. It recomputes rule scores, evidence metrics, and paired
statistics from retained artifacts and rejects hash, schedule, prompt,
configuration, or implementation drift. The committed result returns 15/15
complete pairs, zero errors, and zero warnings.
