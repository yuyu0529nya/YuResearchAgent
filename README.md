<div align="center">

# YuResearchAgent

### 从复杂 Query 到可验证深度研究报告的多智能体系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Async](https://img.shields.io/badge/Async-asyncio-orange.svg)](https://docs.python.org/3/library/asyncio.html)
[![Tests](https://img.shields.io/badge/tests-167%20passing-brightgreen.svg)](tests/unit)
[![CI](https://github.com/yuyu0529nya/YuResearchAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/yuyu0529nya/YuResearchAgent/actions/workflows/ci.yml)

</div>

YuResearchAgent 是一个面向长问题、强证据和可复现评测的 Deep Research Agent。它把复杂研究问题拆成 DAG 子任务，并通过多 Agent 并发检索、共享记忆、上下文压缩、报告合成、对抗审查和统计评测，生成结构化 Markdown 研究报告。

## Highlights

- **多智能体显著优于单轮 LLM**：在 ResearchBench 15 题头对头评测中，Agent 平均综合分 `0.6034` vs 单轮 LLM `0.5586`，相对提升约 **+8.0%**，配对 bootstrap `95% CI=[+0.0134,+0.0761]`，`p=0.0021`，`Cohen's d=0.83`。结果见 [docs/evaluation/headtohead_n15.json](docs/evaluation/headtohead_n15.json)。
- **完整评测体系**：自建 ResearchBench 35 题 × 11 领域，规则指标覆盖事实、引用、幻觉、逻辑和完备性；LLM-as-Judge 用于专家抽查；统计层提供 bootstrap CI、p-value、Cohen's d 和配对 t-test。
- **工程化 Agent 内核**：自研 9 状态 Orchestrator、DAG 拓扑并发、失败重规划、全局超时降级、AgentPool 复用、多后端模型路由。
- **质量与鲁棒性优化**：引用质量从 `4/10` 提升到 `7/10`，LLM-Judge overall 从 `6/10` 提升到 `8/10`；合成器不再只看到 `[Result N]`，而是接收标题/作者/年份/链接结构化来源并生成规范参考文献表。统一 JSON fallback 解析器把畸形 LLM 输出恢复率从 `1/9` 提升到 `9/9`；修复上下文截断、工具失败误判、路径沙箱、统计退化输入等真实问题。
- **真实 GRPO 训练闭环**：在单卡 RTX 5090 上完成 4 组 TRL + LoRA + GRPO 训练实验，覆盖 Qwen2.5-7B-Instruct、Qwen2.5-1.5B-Instruct 和 Qwen2.5-1.5B base，打通 rollout → reward → gradient update → LoRA adapter → held-out eval 的完整链路。
- **可用界面**：提供 CLI、REPL 和 Gradio 流式 Web UI；Web UI 实时展示 planning → dispatching → synthesizing → done 的状态机进度，长任务不再黑盒等待。

## Architecture

```raw
User Query
    |
    v
Planner: JSON DAG decomposition
    |
    v
Orchestrator: 9-state async scheduler + replan
    |
    v
Worker Agents: web / paper / browser / file / calculator / sandbox tools
    |
    v
Memory + Compressor: SQLite vector memory + semantic context compression
    |
    v
Summarizer: cited Markdown report
    |
    v
Optional Red-Blue Adversarial Review
    |
    v
Evaluation: rules + LLM judge + statistical significance
```

| 模块 | 职责 | 关键实现 |
|---|---|---|
| M1 Orchestrator | 多 Agent 调度 | 9 状态状态机、DAG 层级并发、Semaphore、replan、全局超时 |
| M2 Planner | 复杂问题拆解 | LLM 生成 JSON DAG，强约束子任务相关性和依赖合法性 |
| M3 Compressor | 长上下文压缩 | Embedding 粗筛、TextRank 关键句、query-biased 过滤 |
| M4 Memory Store | 跨 Agent 记忆 | SQLite + numpy 向量索引、session 隔离、去重、矛盾检测 |
| M5 Adversarial Loop | Red-Blue 审查 | 五维 Red verdict、Blue 定点编辑、self-verify、截断保护 |
| M6 Evolution | 自进化实验 | GRPO 数据/训练接口 + 独立 TRL/LoRA 真训练 PoC |
| Evaluation | 质量度量 | ResearchBench、HotpotQA variant、规则指标、LLM-Judge、bootstrap |

## Evaluation Results

### Agent vs Single-Shot LLM

`scripts/run_headtohead.py` 用同一批 ResearchBench 题目比较完整 Agent 流程和单轮 LLM 直答，使用规则综合分做配对统计。

| Setting | Agent | Single-shot LLM | Delta |
|---|---:|---:|---:|
| ResearchBench n=15 | 0.6034 | 0.5586 | +0.0448 |

Statistical test:

- `95% CI = [+0.0134, +0.0761]`
- `p = 0.0021`
- `Cohen's d = 0.83`
- Relative lift over baseline: about `+8.0%`

Raw summary: [docs/evaluation/headtohead_n15.json](docs/evaluation/headtohead_n15.json).

### Rule Metrics + LLM-as-Judge

The evaluation stack is deliberately two-layered:

- **Rule-based metrics** are cheap and reproducible: semantic/factual coverage, hallucination risk, citation coverage, logical consistency, comprehensiveness, efficiency.
- **LLM-as-Judge** is used for expert-style audit: factuality, logic, citation quality and confidence, then aggregated with rule scores when needed.

This avoids relying only on subjective judge output while still catching qualitative issues that string rules miss.

### Citation Quality Validation

The strongest recent quality work is a citation upgrade: structured source metadata flows into
the synthesis prompt, so the final report can cite concrete papers and generate a normalized
bibliography with title, author, year, and URL provenance.

| Version | Citation quality | Overall Judge score | Change |
|---|---:|---:|---|
| Baseline | 4/10 | 6/10 | generic `[Result N]` references |
| v1 | 5/10 | - | researcher prompt prefers academic sources |
| v2 | 6/10 | 7/10 | summarizer asks for real source citations |
| v3 | 7/10 | 8/10 | structured source list with title/author/year/link |

Compact evidence: [docs/evaluation/citation_quality_v3.json](docs/evaluation/citation_quality_v3.json).

### GRPO Training System

The project includes a real GRPO training pipeline on RTX 5090: rollout collection, reward scoring,
LoRA updates, adapter export, and held-out evaluation. The experiments were used to validate the
training infrastructure and compare how model size, initialization, and headroom affect observed
RL gains.

| Experiment | Model | Setup | Baseline | After GRPO | Finding |
|---|---|---|---:|---:|---|
| 1 | Qwen2.5-7B-Instruct | LoRA, 400 steps | 89% | 91% | Full 7B rollout/update/eval loop |
| 2 | Qwen2.5-1.5B-Instruct | LoRA, 400 steps | 66.6% | 69.4% | Measured held-out lift at larger n |
| 3 | Qwen2.5-1.5B-Instruct | tuned accumulation/temp/500 steps | 66.6% | 69.4% | Reward and eval dynamics audit |
| 4 | Qwen2.5-1.5B base | R1-Zero-style cold start | 55% | 60% | Larger headroom produces larger lift |

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) and [scripts/grpo_poc/SUMMARY.md](scripts/grpo_poc/SUMMARY.md).

## Engineering Work

- **Robust JSON parser**: one shared parser handles markdown fences, trailing commas, line comments, balanced braces and noisy prefixes. Tests quantify `json.loads` baseline `1/9` vs fallback parser `9/9`.
- **Citation quality**: researcher and summarizer prompts force source title + author/org + year, prefer academic/primary sources for technical tasks, and feed numbered structured sources into synthesis to avoid generic `[Result N]` references.
- **Blue targeted edits**: adversarial repair no longer rewrites the whole report by default; it applies exact `before -> after` replacements to avoid truncating long reports.
- **Timeout control**: OpenAI-compatible client uses explicit request timeout/retry bounds; adversarial stage is wrapped by remaining global timeout.
- **Memory hygiene**: low-quality greetings/errors are rejected before entering long-term memory; session scoped vector retrieval prevents stale cross-run contamination.
- **File sandboxing**: FileReader uses `Path.resolve()` + `is_relative_to()` instead of string-prefix checks.
- **CI and tests**: 167 unit tests run without API keys or GPU; CI covers Python 3.10-3.13.

## Quick Start

```bash
git clone https://github.com/yuyu0529nya/YuResearchAgent.git
cd YuResearchAgent

uv venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.template .env
# Fill one OpenAI-compatible backend, for example QWEN_API_KEY or GLM_API_KEY.
```

Run one research query:

```bash
python scripts/run_single.py \
  --query "2024-2025年大模型Agent技术趋势与落地案例研究" \
  --config configs/default.yaml
```

Interactive REPL:

```bash
python scripts/run_repl.py
```

Streaming Web UI:

```bash
pip install gradio
python scripts/run_webui.py
# http://localhost:7860
```

Unit tests:

```bash
pip install -r requirements-test.txt
pytest tests/unit -q
```

Head-to-head benchmark:

```bash
python scripts/run_headtohead.py --num_questions 15 --config configs/default.yaml
```

## Repository Structure

```raw
YuResearchAgent/
├── configs/                  # YAML config center
├── src/                      # core source, 11.8k LOC
│   ├── orchestrator/         # M1 scheduler and schemas
│   ├── planner/              # M2 DAG planner
│   ├── compressor/           # M3 context compression
│   ├── memory/               # M4 SQLite + vector memory
│   ├── adversarial/          # M5 Red-Blue loop
│   ├── evolution/            # M6 evolution interfaces
│   ├── agents/               # researcher / summarizer agents
│   ├── models/               # OpenAI-compatible model router
│   ├── tools/                # search, browser, file, calculator, sandbox
│   └── utils/                # JSON parsing, tracing, env
├── evaluation/               # ResearchBench, metrics, reports
├── scripts/                  # CLI, benchmark, Web UI, GRPO PoC
├── docs/                     # summarized experiment evidence
└── tests/                    # 167 unit tests
```

## Tech Stack

| Layer | Stack |
|---|---|
| Language | Python 3.10+ |
| Concurrency | asyncio, Semaphore, background thread for Web UI streaming |
| LLM Backends | Qwen / GLM / DeepSeek / MiMo / OpenAI / vLLM-compatible APIs |
| Retrieval | Web search, browser extraction, ArXiv/Semantic Scholar/OpenAlex |
| Memory | SQLite, numpy vector index, sentence-transformers |
| Training PoC | TRL GRPOTrainer, LoRA, PEFT, RTX 5090 |
| Evaluation | ResearchBench, HotpotQA variant, bootstrap, Cohen's d, LLM-as-Judge |
| UI | CLI, REPL, Gradio streaming Web UI |

## Resume Bullets

- Built a deep-research multi-agent system with DAG planning, async orchestration, shared vector memory, citation-aware synthesis, adversarial review, and statistical evaluation.
- Demonstrated significant lift over single-shot LLM baseline on ResearchBench (`+8.0%`, n=15, `p=0.0021`, `d=0.83`) using paired bootstrap testing.
- Designed a reproducible evaluation stack combining rule-based metrics, LLM-as-Judge audit, ResearchBench 35-question suite, and head-to-head benchmarking.
- Improved citation quality from `4/10` to `7/10` and overall Judge score from `6/10` to `8/10` by passing structured source metadata into the synthesis prompt and producing normalized references.
- Implemented and analyzed 4 real GRPO training runs on RTX 5090 with TRL + LoRA across 7B/1.5B/base models, covering rollout collection, reward scoring, LoRA updates, adapter export, and held-out evaluation.
- Hardened production reliability with robust JSON parsing, timeout boundaries, memory quality filters, path sandboxing, targeted report repair, and 167 API-free unit tests in CI.

## License

[MIT](LICENSE) © 2025 YuResearchAgent Contributors.
