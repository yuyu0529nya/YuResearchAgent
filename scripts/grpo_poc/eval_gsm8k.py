#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K 准确率评测 —— 训练前/后对比，产出"GRPO 提分"的硬证据。

用法：
    # 基线（原始模型）
    python eval_gsm8k.py --model Qwen/Qwen2.5-1.5B-Instruct --n 200
    # 训练后（加载 LoRA adapter）
    python eval_gsm8k.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/grpo_qwen1.5b --n 200

两次准确率之差即为 GRPO 的真实增益。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward import extract_final_number  # noqa: E402
from train_grpo import FEWSHOT_BASE_PROMPT, SYSTEM_PROMPT  # noqa: E402


def _nums_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return str(a).strip() == str(b).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 目录（训练后评测时给）")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--base", action="store_true", help="base 模型：用 few-shot 纯文本 prompt")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[eval] 已加载 LoRA adapter: {args.adapter}")
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    correct = 0
    for i, ex in enumerate(ds):
        if args.base:
            prompt = FEWSHOT_BASE_PROMPT.replace("{question}", ex["question"])
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_final_number(gen)
        gold = extract_final_number(ex["answer"])
        if _nums_equal(pred, gold):
            correct += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{args.n}] 累计准确率 {correct/(i+1):.3f}")

    acc = correct / len(ds)
    tag = f"adapter={args.adapter}" if args.adapter else "baseline"
    print(f"\n==== GSM8K 准确率 ({tag}) = {acc:.4f}  ({correct}/{len(ds)}) ====")


if __name__ == "__main__":
    main()
