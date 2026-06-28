#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/metrics/stats.py
================================================================================
统计显著性检验工具：bootstrap 置信区间、效应量、配对 t 检验。

适用于小样本消融实验和 head-to-head benchmark 的统计严谨性验证。

设计说明：
  - bootstrap 支持可选 ``seed``，保证实验结果**可复现**（默认 None 保持随机）。
  - 重采样已向量化（一次性生成 (n_bootstrap, n) 的索引矩阵），相比 Python 循环
    调用 np.random.choice 有数量级的加速。
  - cohens_d 对样本数 < 2 的退化输入返回 0.0，避免 ddof=1 方差除零得到 NaN。
  - paired_t_test 在缺少 scipy 时退化为"手算 t 值 + 正态近似 p"，且返回**与
    scipy 路径一致的字段结构**，避免调用方 KeyError。
================================================================================
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def bootstrap_ci_paired(
    diffs: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    配对差异的 bootstrap 置信区间。

    Args:
        diffs: 配对差异列表（如 full_score - no_adv_score）
        n_bootstrap: bootstrap 采样次数
        confidence: 置信水平
        seed: 随机种子；指定后结果可复现

    Returns:
        dict with mean_diff, ci_lower, ci_upper, p_value, significant, n
    """
    if not diffs:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "significant": False, "n": 0}

    diffs_arr = np.asarray(diffs, dtype=float)
    n = len(diffs_arr)
    mean_diff = float(np.mean(diffs_arr))

    # 向量化重采样：一次生成 (n_bootstrap, n) 索引矩阵，按行求均值
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = diffs_arr[idx].mean(axis=1)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_means, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha / 2) * 100))

    # 单侧 bootstrap p-value：H0 mean_diff <= 0
    p_value = float(np.mean(boot_means <= 0))

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
        "significant": ci_lower > 0,  # 95% CI 完全在 0 右侧
        "n": n,
    }


def bootstrap_ci_two_sample(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    两组独立样本的 bootstrap 置信区间（非配对）。

    Args:
        scores_a: 系统 A 的分数列表
        scores_b: 系统 B 的分数列表
        n_bootstrap: bootstrap 采样次数
        confidence: 置信水平
        seed: 随机种子；指定后结果可复现
    """
    if not scores_a or not scores_b:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "significant": False}

    a_arr = np.asarray(scores_a, dtype=float)
    b_arr = np.asarray(scores_b, dtype=float)
    na, nb = len(a_arr), len(b_arr)
    mean_diff = float(np.mean(a_arr) - np.mean(b_arr))

    rng = np.random.default_rng(seed)
    ia = rng.integers(0, na, size=(n_bootstrap, na))
    ib = rng.integers(0, nb, size=(n_bootstrap, nb))
    boot_diffs = a_arr[ia].mean(axis=1) - b_arr[ib].mean(axis=1)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_diffs, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_diffs, (1 - alpha / 2) * 100))
    p_value = float(np.mean(boot_diffs <= 0))

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
        "significant": ci_lower > 0,
        "n_a": na,
        "n_b": nb,
    }


def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """计算 Cohen's d 效应量。

    样本数不足（任一组 < 2）时无法估计方差，返回 0.0（而非 NaN）。
    """
    a_arr = np.asarray(scores_a, dtype=float)
    b_arr = np.asarray(scores_b, dtype=float)
    if a_arr.size < 2 or b_arr.size < 2:
        return 0.0
    pooled_std = math.sqrt((np.var(a_arr, ddof=1) + np.var(b_arr, ddof=1)) / 2)
    if pooled_std < 1e-9:
        return 0.0
    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled_std)


def paired_t_test(scores_a: list[float], scores_b: list[float]) -> dict[str, Any]:
    """配对 t 检验（假设差值近似正态）。作为 bootstrap 的补充。

    返回字段在有无 scipy 时保持一致：t_statistic, p_value, mean_diff, n, method。
    无 scipy 时手算 t 值，并用正态近似（erfc）给出双侧 p 值。
    """
    a_arr = np.asarray(scores_a, dtype=float)
    b_arr = np.asarray(scores_b, dtype=float)
    diffs = a_arr - b_arr
    n = len(diffs)
    mean_diff = float(np.mean(diffs)) if n else 0.0
    base = {"mean_diff": round(mean_diff, 4), "n": n}

    if n < 2:
        return {**base, "t_statistic": 0.0, "p_value": 1.0, "method": "insufficient_n"}

    try:
        from scipy import stats

        t_stat, p_value = stats.ttest_1samp(diffs, popmean=0)
        return {
            **base,
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 4),
            "method": "scipy_ttest_1samp",
        }
    except ImportError:
        sd = float(np.std(diffs, ddof=1))
        if sd < 1e-9:
            return {**base, "t_statistic": 0.0, "p_value": 1.0, "method": "degenerate_zero_variance"}
        t_stat = mean_diff / (sd / math.sqrt(n))
        # 正态近似双侧 p 值：p = erfc(|t| / sqrt(2))（大样本下接近 t 分布）
        p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
        return {
            **base,
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "method": "normal_approx_no_scipy",
        }
