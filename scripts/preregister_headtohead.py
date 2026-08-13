#!/usr/bin/env python3
"""Freeze the n=15 head-to-head protocol before any outcome is observed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.preregistration import build_preregistration, evaluation_config_snapshot
from evaluation.protocol import atomic_write_json
from evaluation.runtime_identity import (
    configure_single_backend,
    create_policy,
    effective_module_policies,
    policy_identity,
)
from src.core.runner import load_config
from src.models.model_router import ModelRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a frozen head-to-head manifest")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--judge-backend", default="deepseek")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Freeze the shared Agent/baseline/Judge knowledge cutoff (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        dest="question_ids",
        help="Freeze only the named ResearchBench question; repeat for multiple IDs.",
    )
    parser.add_argument(
        "--output",
        default="docs/evaluation/headtohead_v4_preregistration.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    backend = str(args.backend or config.get("model", {}).get("backend", "kimi"))
    configure_single_backend(config, backend)
    module_policies = effective_module_policies(config, backend)
    baseline = create_policy(config, backend, "summarizer")
    judge = ModelRouter.create_backend(args.judge_backend, use_cache=False)
    manifest = build_preregistration(
        backend=backend,
        model_name=baseline.model_name,
        sampling=policy_identity(baseline),
        judge_backend=args.judge_backend,
        judge_model_name=judge.model_name,
        judge_sampling=policy_identity(judge),
        question_ids=args.question_ids,
        seed=args.seed,
        as_of_date=args.as_of_date,
        agent_configuration=evaluation_config_snapshot(
            config,
            backend,
            effective_module_policies=module_policies,
        ),
    )
    atomic_write_json(args.output, manifest)
    print(
        f"Frozen {len(manifest['question_ids'])} questions across "
        f"{len(manifest['domain_counts'])} domains: {args.output}\n"
        f"SHA-256: {manifest['fingerprint_sha256']}"
    )


if __name__ == "__main__":
    main()
