# v6 Development Coverage Diagnostic

This is a three-question development diagnostic, not a confirmatory benchmark
and not a replacement for the preregistered v5 result. The protocol was frozen
in `docs/evaluation/headtohead_v6_dev_preregistration.json` before execution.

## Result

| Measure (`n=3`) | Agent | One-call baseline | Difference |
|---|---:|---:|---:|
| Rule composite | 0.6526 | 0.6024 | +0.0502 |
| Counterbalanced Judge (1-5) | 4.0417 | 4.2500 | -0.2083 |
| Mean agent task coverage | 58.3% | n/a | n/a |

The exact paired sign-flip test for the rule composite is `p=0.125`; with only
three pairs, no significance claim is warranted. The Judge result is directionally
unfavorable and rules out presenting the coverage-contract change as an answer-
quality improvement.

## Diagnosis

The new deterministic task-coverage audit found no `synthesis_gap`: when a task
had a usable source, the final report addressed it. The missing dimensions were
all `research_gap`s caused by task timeouts or no usable source. Strict final
claim support remained weak (mean 5.52%), while source quality still favored the
Agent. The next intervention therefore targets per-task deadline allocation and
evidence acquisition, not report prose.

`result.json` retains paired metrics, raw counterbalanced Judge decisions,
telemetry, task-coverage records, and hashes. `reports/` contains every generated
Agent/baseline report and copied evidence artifact.
