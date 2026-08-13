# Head-to-Head v5 Artifact

This directory contains the complete output of the preregistered v5 comparison:
full YuResearchAgent versus one Kimi K3 call on the same 15 questions, followed
by two counterbalanced DeepSeek Judge evaluations per pair.

## Files

- `result.raw.json`: byte-for-byte runner output. Its report paths point to the
  original ignored `outputs/evaluation/` tree.
- `result.json`: portable copy with only report/evidence paths rewritten to this
  committed directory. Scores, reports, hashes, telemetry, and Judge data are
  unchanged.
- `reports/*_agent.md`: 15 exact Agent reports.
- `reports/*_baseline.md`: 15 exact one-call baseline reports.
- `reports/evidence/*.json`: 15 exact final EvidenceGraph artifacts.

## Audit

From the repository root:

```bash
python scripts/audit_headtohead.py \
  docs/evaluation/artifacts/headtohead_v5/result.json \
  --preregistration docs/evaluation/headtohead_v5_preregistration.json \
  --require-complete
```

Expected status:

```json
{
  "valid": true,
  "complete": true,
  "completed_pairs": 15,
  "expected_pairs": 15,
  "errors": [],
  "warnings": []
}
```

## Interpretation

The preregistered rule composite improved by `+0.0629` (95% bootstrap CI
`[+0.0452, +0.0828]`, exact one-sided paired sign-flip `p=0.000061`, paired
`d_z=1.6477`). The independent Judge endpoint was tied within uncertainty:
`-0.0417/5`, 95% CI `[-0.4417, +0.3667]`, `p=0.593018`.

The Agent substantially improved citation coverage, but strict claim-support
coverage was only `5.79%`, and the Agent used about `26.4x` the generation tokens
and `3.28x` the wall time. See `docs/EXPERIMENTS.md` for the complete claim and
limitation ledger.
