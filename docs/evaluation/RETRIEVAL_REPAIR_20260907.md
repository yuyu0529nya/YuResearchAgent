# Retrieval Repair: 2026-09-07

Status: development diagnostics, not a benchmark or evidence of superiority.

## Initial Observation

Question: 比较 RAG 与长上下文方法在企业知识库问答中的主要优缺点，并给出可验证的来源

The initial DeepSeek V4 Pro run took 236.03 seconds and made 16 tool calls.
The retained evidence artifact contained 21 sources, all abstract-only, with
0/48 claims supported. The final semantic verifier reviewed only 3 of 31
eligible claims. Unreviewed lexical NEI labels are not factual refutations.
The original run did not retain per-tool diagnostics or provider usage in its
sidecar, so its exact browser failures cannot be reconstructed from the log.

The preceding single-agent diagnostic had a different tool/time budget. It is
not a matched-budget comparison, and neither diagnostic supports significance
claims or a project quality score.

## Reproduced Problems and Repairs

- Search silently defaulted to GPT-4.1 mini even when the main model was
  DeepSeek. That route returned a region error in the original diagnostic.
  A standalone DeepSeek search succeeded and returned directly relevant
  RAG/long-context comparison papers. Search now inherits `OPENROUTER_MODEL`
  unless `OPENROUTER_SEARCH_MODEL` explicitly overrides it.
- Browser reads were reserved until near budget exhaustion. A failed first
  document could therefore leave no recovery slot. Reads are now scheduled
  earlier, and failed/empty retrieval no longer requests premature synthesis.
- Curl rejected redirects; aiohttp could accept redirect bodies as content.
  Both transports now follow a bounded, explicitly validated redirect chain.
  This does not claim comprehensive DNS-rebinding protection for public hosting.
- arXiv abstract URLs did not try PDF after an unavailable HTML conversion;
  requested version suffixes were also discarded. PDF fallback now precedes
  the explicitly labeled abstract fallback, preserving the requested version.
- Paper selection, context matching, and evidence ingestion disagreed on
  whether to prefer the PDF or landing URL. All now prefer the paper's PDF URL.
- The synthesizer received at most 360 characters from a source excerpt.
  The per-source excerpt budget is now 1200 characters, still bounded.
- The CLI discarded runtime metadata. It now saves a JSON sidecar with provider
  usage, tool arguments/failures, and the exact evidence artifact path.
- Worker generation now defaults to a 2400-token cap. This is a latency guard
  for the DeepSeek V4 Pro route; the synthesizer keeps its larger report budget.
- OpenRouter search now falls back to curl when the configured proxy can reach
  the API but aiohttp cannot complete TLS negotiation. The fallback preserves
  the same request and does not expose credentials in diagnostics.

The current [OpenRouter documentation](https://openrouter.ai/docs/guides/features/plugins/web-search)
documents model-agnostic search and source annotations. A model supporting the
API is still subject to provider availability; the local live probe is the
evidence that the selected DeepSeek route worked here.

## Limits

The fast runtime still caps semantic verification at three ambiguous claims.
An incomplete audit intentionally does not trigger automatic evidence-based
rewriting, because unreviewed NEI cannot justify deleting claims. More complete
semantic auditing and multilingual evidence retrieval remain separate work.
No support threshold was lowered, and no previous confirmatory result changed.

Generated reports and sidecars are local diagnostics. Their paths and hashes are
recorded in [retrieval_repair_manifest.json](retrieval_repair_manifest.json).
