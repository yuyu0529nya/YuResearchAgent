# Algorithm and Paper Mapping

This document separates implemented behavior from paper inspiration. The project
does not claim to reproduce a paper unless its model, training objective, and
evaluation protocol are actually present.

## Adoption Matrix

| Paper / system | Relevant idea | YuResearchAgent mapping | Adoption level |
|---|---|---|---|
| [CAGE: Cognitive Attribution Graphs](https://arxiv.org/abs/2607.24236) | Explicit claim-document attribution before long-form generation | Typed `ClaimRecord -> EvidenceChunk -> SourceRecord` graph; synthesis receives claim verdicts and exact source order | Independent implementation of the graph concept; no CAGE induction/reasoning model training |
| [WebWeaver](https://arxiv.org/abs/2509.13312) | Interleave evidence acquisition with plan/outline refinement | Initial DAG, post-collection completeness audit, unresolved-claim `VERIFY` round | Partial; no section-level learned outline controller |
| [A-RAG](https://arxiv.org/abs/2602.03442) | Expose retrieval at several granularities | Search results, academic metadata, full-page reading, evidence chunks | Partial; tool choice is LLM-driven, but there is no learned action policy |
| [FS-Researcher](https://arxiv.org/abs/2602.01566) | Durable workspace beyond the context window | SQLite memory plus JSON evidence artifacts with stable IDs and hashes | Partial; artifacts are durable, but workers do not navigate a hierarchical file workspace |
| [ReSum](https://arxiv.org/abs/2509.13313) | Compress long trajectories into reusable reasoning state | Budget-triggered relevance filtering, TextRank extraction, and LLM summary before synthesis | Partial; periodic trajectory checkpointing and ReSum-GRPO are not implemented |
| [DeepResearch Bench II](https://arxiv.org/abs/2601.08536) | Atomic, verifiable report diagnostics | Canonical factual metric, claim-level audit, separate rule/Judge layers | Methodological influence; the local 35-question suite is not benchmark-equivalent |

## 1. Claim-Evidence-Source Graph

Code: `src/evidence/schemas.py`, `src/evidence/store.py`, and
`src/evidence/verifier.py`.

Tool trajectories are normalized into three typed entities:

```text
SourceRecord 1 --- N EvidenceChunk N --- N ClaimRecord
```

- `SourceRecord` stores URL, title, author/year, source type, quality metadata,
  task provenance, and a hash of the full captured document.
- `EvidenceChunk` stores the exact text used for verification, its kind
  (`search_snippet`, `abstract`, `full_text`, or `file`), locator, task ID, and
  its own hash.
- `ClaimRecord` stores citations, candidate/support/contradiction edges, a
  three-way verdict, score, and explanation.

Source and evidence IDs are SHA-256-derived and deterministic. URL normalization
and source-key deduplication prevent the same page from inflating source counts.

## 2. Deterministic Verification Pass

The verifier first extracts sentence-sized claims, excluding headings,
bibliographies, questions, and confidence boilerplate. For a claim `c` and
evidence chunk `e`, the deterministic candidate score is:

```text
score(c, e) = token_coverage(c, e)
              * (0.75 + 0.25 * source_quality(e))
              * evidence_kind_weight(e)
```

`evidence_kind_weight` is `0.82` for a search snippet, `0.95` for an abstract,
and `1.0` for full text or a local file. This intentionally makes snippets less
persuasive than opened documents.

The verdict also checks:

- numeric compatibility, including decimals/thousands separators damaged by
  HTML or PDF whitespace;
- polarity conflicts such as `supports` versus `does not support`;
- final-report citation constraints, using the exact source ordering passed to
  the synthesizer.

The fallback label is always `NOT_ENOUGH_EVIDENCE`. In `hybrid` mode, only
ambiguous pairs are sent to a strict LLM verifier, which receives no outside
context and may only use the supplied excerpts. Verification requests are
batched in groups of at most eight claims so the serialized evidence remains
below the model wrapper's truncation boundary.

This is deliberately conservative. The score is a reproducible engineering
heuristic, not a natural-language-inference probability.

## 3. Evidence-Gap Planning

Code: `src/evidence/gap_planner.py` and
`src/orchestrator/orchestrator.py`.

After workers finish, the orchestrator audits their outputs. If coverage is below
`evidence.min_coverage`, unresolved claims are ranked by:

1. contradicted claims before merely unsupported claims;
2. claims containing exact numeric details;
3. lower existing verification score.

The top claims become bounded `VERIFY` tasks. Their prompt requires an original
paper, official document, or institutional source and at least one opened
full-text page. New trajectories are ingested into the same store, then all
successful results are synthesized together.

This gives the system a closed loop:

```text
research -> ingest -> verify -> identify gaps -> targeted research -> synthesize
```

It differs from WebWeaver because the revision target is claim support rather
than a learned report outline.

## 4. Evidence-Constrained Synthesis

Code: `src/agents/summarizer.py`.

The synthesizer receives:

- worker findings;
- an ordered source catalog with title, author, year, URL, and source ID;
- supported, refuted, and unresolved claims with evidence excerpts.

Supported claims may be stated with mapped citation numbers. Refuted claims must
not be repeated as facts. Unresolved claims must be qualified or omitted. A
final audit parses the generated report again with the exact bibliography order,
and report confidence is multiplied by an evidence-coverage factor.

## 5. Long-Context Control

Code: `src/compressor/` and `src/memory/embedder.py`.

Synthesis input is copied before compression so raw tool evidence remains intact.
Compression is budget-triggered:

- L1: query-biased relevance filtering;
- L2: TextRank-style extractive sentence selection;
- L3: per-document and aggregate LLM summaries;
- fallback: bounded sliding window.

When a sentence-transformer is unavailable, the embedder uses normalized feature
hashing over English words/bigrams and Chinese 2/3-grams. It preserves lexical
similarity and is deterministic; it does not pretend to provide semantic model
quality.

## 6. Evaluation Algorithms

Code: `evaluation/metrics/`, `evaluation/report_sampling.py`, and
`evaluation/benchmarks/research_bench.py`.

The current ResearchBench composite is:

```text
0.25 factual_accuracy
+ 0.20 logical_consistency
+ 0.20 citation_coverage
+ 0.20 bias
+ 0.15 comprehensiveness
```

`factual_accuracy` is explicitly emitted as `0.4 * string_accuracy + 0.6 *
semantic_accuracy`. Legacy metric dictionaries are normalized at the scoring
boundary. LLM-as-Judge receives balanced excerpts from the beginning, middle,
end, and bibliography rather than a prefix-only truncation.

Module ablations execute real configuration switches and require an evaluator
for quality scores. Execution success is reported separately; an absent evaluator
never produces a synthetic score of `1.0`.

Exact report/evidence pairs can be re-audited with `scripts/replay_evidence.py`.
The loader verifies every retained evidence chunk against its content hash and
the CLI can pin the SHA-256 of both input files before recomputing verdicts.

## What Is Not Yet Implemented

- A learned retrieval policy or test-time compute controller.
- CAGE's trained cognitive-map induction and structured citation model.
- WebWeaver's continuously revised, section-level dynamic outline.
- ReSum-GRPO or an online update from research trajectories to the live policy.
- DeepResearch Bench II's 9,430 expert-reviewed binary rubrics.
- A statistically valid rerun of Agent versus single-shot under the corrected
  ResearchBench v2 metric contract.

These boundaries are intentional: they define the next experiments instead of
being hidden behind architecture labels.
