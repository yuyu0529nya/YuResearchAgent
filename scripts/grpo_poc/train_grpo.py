#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO PoC 训练脚本 —— 单卡（RTX 5090 / 32GB）即可跑。

用 TRL 的 GRPOTrainer 在 GSM8K 上对 Qwen2.5-1.5B-Instruct 做 LoRA + GRPO 训练，
可验证奖励（答案对错）驱动，产出"准确率随训练上升"的真实证据。

这是 M6 自进化引擎里"GRPO 梯度更新"那一步的独立、最小、可运行实证：
M6 负责出题/收集轨迹/塑形 reward/导出数据，本脚本验证其指向的 GRPO trainer 真能跑通。

快速校验管线（几分钟，不求效果）：
    python train_grpo.py --smoke
正式训练：
    python train_grpo.py --max_steps 300 --output_dir outputs/grpo_qwen1.5b
依赖见 requirements.txt；5090(Blackwell) 需 CUDA 12.8 的 torch 轮子，见 README。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward import correctness_reward, format_reward  # noqa: E402

SYSTEM_PROMPT = (
    "你是严谨的数学解题助手。请逐步推理，"
    "并在最后一行用 '#### <最终数字答案>' 的格式给出答案。"
)

# base（非 instruct）模型不跟随 chat 指令，用 few-shot 纯文本格式冷启动：
# 2 个范例教会"逐步推理 + #### 答案"，GRPO 再优化正确率（R1-Zero 式）。
FEWSHOT_BASE_PROMPT = (
    'Solve the math problem step by step. End your answer with "#### <final number>".\n\n'
    "Question: Natalia sold clips to 48 friends in April, then half as many in May. "
    "How many clips did she sell altogether?\n"
    "Answer: In April she sold 48. In May she sold 48 / 2 = 24. "
    "Altogether 48 + 24 = 72. #### 72\n\n"
    "Question: Weng earns $12 per hour babysitting. Yesterday she babysat 50 minutes. "
    "How much did she earn?\n"
    "Answer: Per minute she earns 12 / 60 = 0.2 dollars. For 50 minutes: 0.2 * 50 = 10. #### 10\n\n"
    "Question: {question}\nAnswer:"
)


def build_dataset(split: str, n: int | None, base: bool = False):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    if n:
        ds = ds.select(range(min(n, len(ds))))

    def _fmt(ex):
        if base:
            # 纯文本 prompt（GRPOTrainer 接受 string 形式的 prompt）
            return {"prompt": FEWSHOT_BASE_PROMPT.replace("{question}", ex["question"]), "answer": ex["answer"]}
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "answer": ex["answer"],
        }

    return ds.map(_fmt, remove_columns=[c for c in ds.column_names if c != "answer"])


def main():
    ap = argparse.ArgumentParser(description="GRPO PoC on GSM8K (single GPU)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--output_dir", default="outputs/grpo_qwen1.5b")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=8, help="per-device prompts/step")
    ap.add_argument("--num_generations", type=int, default=8, help="GRPO 组大小 G")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数=每次更新用多少道不同题")
    ap.add_argument("--temperature", type=float, default=1.0, help="采样温度，高→completion 更多样，减少零优势")
    ap.add_argument("--base", action="store_true", help="base(非instruct)模型：用 few-shot 纯文本 prompt")
    ap.add_argument("--max_prompt_len", type=int, default=400)
    ap.add_argument("--max_completion_len", type=int, default=512)
    ap.add_argument("--use_vllm", action="store_true", help="用 vLLM 加速采样(Blackwell 需新版本)")
    ap.add_argument("--smoke", action="store_true", help="极小规模冒烟，仅验证管线能跑")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    n_train = 64 if args.smoke else None
    train_ds = build_dataset("train", n_train, base=args.base)
    print(f"[data] 训练样本数: {len(train_ds)}")

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        # 注：trl 1.6+ 的 GRPOConfig 移除了 max_prompt_length（提示词不再按此截断）；
        # GSM8K 提示词很短，无需截断，故仅设置 completion 长度。
        max_completion_length=args.max_completion_len,
        temperature=args.temperature,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=50 if not args.smoke else 5,
        max_steps=8 if args.smoke else args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=args.use_vllm,
        report_to="none",
        log_completions=True,
    )

    # 显式整模加载到单卡：不用 device_map="auto"，否则会把部分层 offload 到 CPU/meta，
    # 导致反向传播时 "expected device meta but got cuda:0"。low_cpu_mem_usage=False 杜绝 meta 残留。
    print(f"[model] 加载 {args.model}（bf16, 单卡, 无 device_map）……")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False  # 与 gradient_checkpointing 兼容

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, format_reward],
        args=cfg,
        train_dataset=train_ds,
        peft_config=lora,
    )

    print("[train] 开始 GRPO 训练……（关注日志里的 reward / rewards/correctness_reward 是否上升）")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[done] LoRA adapter 已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
