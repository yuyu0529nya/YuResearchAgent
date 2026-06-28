#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_headtohead.py
================================================================================
Agent vs 单轮 LLM 头对头评测（客观规则打分 + 统计显著性）。

设计：
    - 同一批 ResearchBench 题目，分别用「完整 Agent 流程」与「单轮 LLM 直答」生成报告；
    - 两份报告都用 ResearchBench 的 ground_truth/expected_topics 做**规则打分**
      （无 LLM-judge 偏置，客观可复现）；
    - 对配对的综合分差异做 Bootstrap 95% CI + Cohen's d，给出是否统计显著。

用法：
    python scripts/run_headtohead.py --num_questions 3 --config configs/default.yaml
================================================================================
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runner import initialize_modules, load_config, run_research, setup_logging
from src.models.model_router import ModelRouter
from evaluation.benchmarks.research_bench import ResearchBench
from evaluation.metrics.stats import bootstrap_ci_paired, cohens_d

_BASELINE_SYSTEM = (
    "你是一位研究助手。请针对用户的问题，直接写一份**全面、结构化**的研究报告"
    "（Markdown 格式，包含要点与分析，尽量给出来源链接）。结尾给出 Overall Confidence: X.XX"
)


def run_baseline(query: str, backend: str) -> str:
    """单轮 LLM 基线：直接一次调用生成报告（不走 Agent 编排/搜索）。"""
    policy = ModelRouter.create_backend(backend)
    resp = policy([
        {"role": "system", "content": _BASELINE_SYSTEM},
        {"role": "user", "content": query},
    ])
    return resp.get("content", "") or ""


async def main() -> None:
    parser = argparse.ArgumentParser(description="Agent vs 单轮 LLM 头对头评测")
    parser.add_argument("--num_questions", type=int, default=3)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/evaluation/headtohead.json")
    args = parser.parse_args()

    setup_logging("INFO")
    cfg = load_config(args.config)
    backend = cfg.get("model", {}).get("backend", "qwen")
    bench = ResearchBench()
    questions = bench.get_questions(n=args.num_questions)
    print(f"[H2H] {len(questions)} 题 | Agent 流程 vs 单轮 {backend} 基线\n")

    rows = []
    for i, q in enumerate(questions, 1):
        qid, query = q["id"], q["query"]
        print(f"[{i}/{len(questions)}] {qid} ...")

        # Agent 完整流程
        modules = initialize_modules(cfg, session_id=f"h2h_{qid}")
        t0 = time.time()
        agent_report = await run_research(query, cfg, modules)
        agent_t = time.time() - t0
        agent_score = bench.evaluate_report(agent_report, qid)["composite_score"]

        # 单轮 LLM 基线
        t0 = time.time()
        base_report = run_baseline(query, backend)
        base_t = time.time() - t0
        base_score = bench.evaluate_report(base_report, qid)["composite_score"]

        rows.append({"qid": qid, "agent": agent_score, "baseline": base_score})
        print(f"    Agent={agent_score:.3f}({agent_t:.0f}s)  Baseline={base_score:.3f}({base_t:.0f}s)  "
              f"Δ={agent_score - base_score:+.3f}")

    a = [r["agent"] for r in rows]
    b = [r["baseline"] for r in rows]
    diffs = [x - y for x, y in zip(a, b)]
    ci = bootstrap_ci_paired(diffs, seed=42)
    d = cohens_d(a, b)

    print("\n" + "=" * 56)
    print(f" Agent 平均综合分:    {sum(a) / len(a):.3f}")
    print(f" Baseline 平均综合分: {sum(b) / len(b):.3f}")
    print(f" Δ(Agent-Baseline) = {ci['mean_diff']:+.3f}  95%CI=[{ci['ci_lower']:+.3f}, {ci['ci_upper']:+.3f}]")
    print(f" p={ci['p_value']:.4f}  显著={'是' if ci['significant'] else '否'}  Cohen's d={d:.2f}")
    print("=" * 56)

    out = {
        "rows": rows,
        "agent_avg": sum(a) / len(a),
        "baseline_avg": sum(b) / len(b),
        "stats": ci,
        "cohens_d": d,
        "num_questions": len(rows),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[H2H] 已保存: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
