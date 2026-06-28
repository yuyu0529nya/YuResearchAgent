# -*- coding: utf-8 -*-
"""
GRPO PoC — 可验证奖励函数（纯 Python，可在无 GPU 环境单测）。

任务：GSM8K 数学题。模型被要求逐步推理后用 "#### <数字>" 给出最终答案。
奖励设计（两个独立 reward_func，TRL GRPOTrainer 会自动相加）：
  - correctness_reward：最终答案数值正确 → 1.0，否则 0.0（主信号，可验证、不可 hack）
  - format_reward：输出包含规范的 "#### 答案" 标记 → 0.2（轻量塑形，帮助早期学习）

TRL GRPOTrainer 的 reward_func 约定签名：
    func(prompts, completions, **kwargs) -> list[float]
其中 completions 是会话格式 list[list[{"role","content"}]]，
数据集里的其它列（这里是 "answer"）通过 **kwargs 传入。
"""
from __future__ import annotations

import re

__all__ = ["extract_final_number", "correctness_reward", "format_reward"]

_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def extract_final_number(text: str) -> str | None:
    """从文本中抽取最终数字答案。优先取 '#### N'，否则取最后一个数字。"""
    if not text:
        return None
    m = re.search(rf"####\s*({_NUM})", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(_NUM, text)
    return nums[-1].replace(",", "") if nums else None


def _nums_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


def _completion_text(completion) -> str:
    """兼容会话格式 [{"role","content"}] 与纯字符串。"""
    if isinstance(completion, list):
        return completion[-1].get("content", "") if completion else ""
    return str(completion)


def correctness_reward(prompts=None, completions=None, answer=None, **kwargs) -> list[float]:
    """主奖励：最终数值答案正确得 1.0。answer 为 GSM8K 金标（含 '#### N'）。"""
    completions = completions or []
    answers = answer or [None] * len(completions)
    rewards = []
    for comp, gold in zip(completions, answers):
        pred = extract_final_number(_completion_text(comp))
        gold_num = extract_final_number(gold) if gold is not None else None
        rewards.append(1.0 if _nums_equal(pred, gold_num) else 0.0)
    return rewards


def format_reward(prompts=None, completions=None, **kwargs) -> list[float]:
    """塑形奖励：输出含规范 '#### <数字>' 标记得 0.2。"""
    completions = completions or []
    rewards = []
    for comp in completions:
        text = _completion_text(comp)
        rewards.append(0.2 if re.search(rf"####\s*{_NUM}", text) else 0.0)
    return rewards
