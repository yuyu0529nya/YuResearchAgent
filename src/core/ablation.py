#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/ablation.py
================================================================================
消融实验通用框架。

对外接口:
    - AblationStudy.run_module_ablation(config, questions, systems) -> dict
    - AblationStudy.run_rounds_ablation(config, questions, max_rounds) -> dict
================================================================================
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable

from .runner import initialize_modules, run_research

logger = logging.getLogger("ablation")


class AblationStudy:
    """消融实验框架：支持模块消融和对抗轮数消融。"""

    # 模块消融的默认配置映射
    DEFAULT_MODULE_ABLATIONS: dict[str, tuple[str, dict]] = {
        "full": ("完整系统", {}),
        "no_adversarial": ("关闭对抗降噪", {"adversarial": {"enabled": False}}),
        "no_compressor": ("关闭上下文压缩", {"compressor": {"enable_multilevel": False}}),
        "no_memory": ("关闭记忆存储", {"memory": {"enabled": False}}),
        "no_evidence": ("关闭 Claim-Evidence 证据图", {"evidence": {"enabled": False}}),
        "heuristic_verifier": (
            "仅使用确定性启发式证据核验",
            {"evidence": {"verification_mode": "heuristic"}},
        ),
        "no_gap_research": (
            "关闭证据缺口补充检索",
            {"planner": {"enable_completeness_check": False}},
        ),
    }

    @staticmethod
    def override_config(config: dict, overrides: dict) -> dict:
        """深度合并配置覆盖（支持嵌套字典）。"""
        cfg = copy.deepcopy(config)

        def _deep_merge(base: dict, patch: dict) -> dict:
            for key, value in patch.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    base[key] = _deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        return _deep_merge(cfg, overrides)

    # -----------------------------------------------------------------------
    # 模块消融：full / no_XXX
    # -----------------------------------------------------------------------
    @classmethod
    def run_module_ablation(
        cls,
        config: dict,
        questions: list[dict[str, Any]],
        systems: dict[str, tuple[str, dict]] | None = None,
        evaluator: Callable[[str, dict[str, Any]], float | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        运行模块消融实验。

        Args:
            config: 基础配置。
            questions: 评测题目列表（每条含 id, query）。
            systems: 消融配置映射。键为 system_name，值为 (描述, 配置覆盖)。
                     默认使用 DEFAULT_MODULE_ABLATIONS。

        Returns:
            包含各系统得分和明细的字典。
        """
        if systems is None:
            systems = cls.DEFAULT_MODULE_ABLATIONS

        results: list[dict[str, Any]] = []

        for name, (desc, overrides) in systems.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"[消融实验] {name}: {desc}")
            logger.info(f"{'='*60}")

            cfg = cls.override_config(config, overrides)
            modules = initialize_modules(cfg)

            scores: list[float] = []
            details: list[dict[str, Any]] = []
            successful_runs = 0

            for q in questions:
                qid = q.get("id", "unknown")
                query = q.get("query", "")
                logger.info(f"  [{qid}] {query[:60]}...")

                start = time.time()
                try:
                    report = asyncio.run(run_research(query, cfg, modules))
                    elapsed = time.time() - start

                    successful_runs += 1
                    detail = {
                        "question_id": qid,
                        "query": query,
                        "elapsed_seconds": elapsed,
                        "report_length": len(report),
                        "system": name,
                    }
                    if evaluator is not None:
                        evaluation = evaluator(report, q)
                        score = cls._extract_score(evaluation)
                        scores.append(score)
                        detail["composite_score"] = score
                        detail["evaluation"] = evaluation
                    else:
                        detail["composite_score"] = None
                    details.append(detail)
                    logger.info(
                        f"    → 成功, time={elapsed:.1f}s, len={len(report)}, "
                        f"score={detail['composite_score']}"
                    )

                except Exception as e:
                    logger.warning(f"    → 失败: {e}")
                    details.append({
                        "question_id": qid,
                        "query": query,
                        "error": str(e),
                        "system": name,
                    })
                    if evaluator is not None:
                        scores.append(0.0)

            results.append({
                "system_name": name,
                "description": desc,
                "num_questions": len(questions),
                "average_composite_score": sum(scores) / len(scores) if scores else None,
                "execution_success_rate": successful_runs / len(questions) if questions else 0.0,
                "details": details,
            })

        return {
            "evaluation_name": "YuResearchAgent 模块消融实验",
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "systems": results,
            "summary": {r["system_name"]: r["average_composite_score"] for r in results},
        }

    # -----------------------------------------------------------------------
    # 对抗轮数消融：0/1/2/3 轮
    # -----------------------------------------------------------------------
    @classmethod
    def run_rounds_ablation(
        cls,
        config: dict,
        questions: list[dict[str, Any]],
        max_rounds: int = 3,
        evaluator: Callable[[str, dict[str, Any]], float | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        在不同对抗轮数下运行评测。

        Args:
            config: 基础配置。
            questions: 评测题目列表。
            max_rounds: 最大对抗轮数。

        Returns:
            键为 adv_0 / adv_1 / ... / adv_N 的结果字典。
        """
        summary: dict[str, float | None] = {}
        full_details: dict[str, Any] = {}

        for rounds in range(max_rounds + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"正在运行对抗轮数 = {rounds}")
            logger.info(f"{'='*50}")

            overrides = {
                "adversarial": {
                    "max_rounds": rounds,
                    "enabled": rounds > 0,
                }
            }
            cfg = cls.override_config(config, overrides)
            modules = initialize_modules(cfg)

            scores: list[float] = []
            details: list[dict[str, Any]] = []

            for idx, q in enumerate(questions, 1):
                qid = q.get("id", f"q{idx}")
                query = q.get("query", "")
                logger.info(f"  [{idx}/{len(questions)}] {qid}")

                try:
                    report = asyncio.run(run_research(query, cfg, modules))
                    detail = {
                        "question_id": qid,
                        "query": query,
                        "rounds": rounds,
                        "report_length": len(report),
                    }
                    if evaluator is not None:
                        evaluation = evaluator(report, q)
                        score = cls._extract_score(evaluation)
                        scores.append(score)
                        detail["composite_score"] = score
                        detail["evaluation"] = evaluation
                    else:
                        detail["composite_score"] = None
                    details.append(detail)
                except Exception as e:
                    logger.warning(f"    → 失败: {e}")
                    if evaluator is not None:
                        scores.append(0.0)
                    details.append({
                        "question_id": qid,
                        "query": query,
                        "rounds": rounds,
                        "error": str(e),
                    })

            avg_score = sum(scores) / len(scores) if scores else None
            key = f"adv_{rounds}"
            summary[key] = avg_score
            full_details[key] = details
            if avg_score is None:
                logger.info(f"对抗轮数 {rounds} 已运行，但未提供 evaluator，质量分未计算")
            else:
                logger.info(f"对抗轮数 {rounds} 平均得分: {avg_score:.4f}")

        return {
            "evaluation_name": "YuResearchAgent 对抗轮数消融实验",
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "details": full_details,
            "config": config,
        }

    @staticmethod
    def _extract_score(evaluation: float | dict[str, Any]) -> float:
        """Normalize evaluator output and reject ambiguous fake scores."""
        if isinstance(evaluation, (int, float)):
            return float(evaluation)
        if isinstance(evaluation, dict) and "composite_score" in evaluation:
            return float(evaluation["composite_score"])
        raise ValueError("evaluator must return a number or a dict containing 'composite_score'")

    # -----------------------------------------------------------------------
    # 结果保存
    # -----------------------------------------------------------------------
    @staticmethod
    def save_results(data: dict[str, Any], output_dir: str, prefix: str = "ablation") -> str:
        """保存消融结果到 JSON 文件。"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"{prefix}_{timestamp}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"消融结果已保存: {filepath}")
        return filepath
