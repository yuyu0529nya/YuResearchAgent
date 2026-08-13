# Evaluation Artifact Index

This directory separates reproducible evidence from development traces and
superseded results. See `../EXPERIMENTS.md` for interpretation and limitations.

| Artifact | Evidence level | What it establishes |
|---|---|---|
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
