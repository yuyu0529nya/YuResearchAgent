"""
tests/unit/test_grpo_reward.py
GRPO PoC 奖励函数单测（本地无 GPU 即可跑，给 5090 训练前先验证最易踩坑的逻辑）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "grpo_poc"))

from reward import correctness_reward, extract_final_number, format_reward


def test_extract_prefers_hash_marker():
    assert extract_final_number("推理过程……\n#### 42") == "42"
    assert extract_final_number("答案是 1,234 元 #### 1,234") == "1234"


def test_extract_falls_back_to_last_number():
    assert extract_final_number("先算 3，再算 5，所以是 8") == "8"
    assert extract_final_number("没有数字") is None
    assert extract_final_number("") is None


def test_correctness_reward_exact_match():
    comps = [[{"role": "assistant", "content": "推理…\n#### 18"}]]
    golds = ["Janet 卖蛋……\n#### 18"]
    assert correctness_reward(completions=comps, answer=golds) == [1.0]


def test_correctness_reward_wrong_answer():
    comps = [[{"role": "assistant", "content": "#### 99"}]]
    golds = ["#### 18"]
    assert correctness_reward(completions=comps, answer=golds) == [0.0]


def test_correctness_reward_handles_commas_and_floats():
    comps = [[{"role": "assistant", "content": "#### 1,000"}]]
    assert correctness_reward(completions=comps, answer=["#### 1000"]) == [1.0]
    comps2 = [[{"role": "assistant", "content": "#### 3.5"}]]
    assert correctness_reward(completions=comps2, answer=["#### 3.50"]) == [1.0]


def test_correctness_reward_batch():
    comps = [
        [{"role": "assistant", "content": "#### 7"}],
        [{"role": "assistant", "content": "#### 8"}],
    ]
    golds = ["#### 7", "#### 9"]
    assert correctness_reward(completions=comps, answer=golds) == [1.0, 0.0]


def test_format_reward():
    good = [[{"role": "assistant", "content": "推理…\n#### 42"}]]
    bad = [[{"role": "assistant", "content": "答案大概是四十二吧"}]]
    assert format_reward(completions=good) == [0.2]
    assert format_reward(completions=bad) == [0.0]


def test_reward_accepts_plain_string_completion():
    # 兼容非会话格式
    assert correctness_reward(completions=["#### 5"], answer=["#### 5"]) == [1.0]
