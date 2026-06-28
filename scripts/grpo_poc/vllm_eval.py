#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 vLLM 批量评测 GSM8K（快）——一次加载，跑 baseline(基座) + post(基座+LoRA) 两轮，
直接给出准确率、增益和双比例 z 检验。适合在全测试集(1319)上钉死显著性。

用法：
    python vllm_eval.py --model <1.5B路径或HF id> --adapter <lora目录> --n 0   # n=0 全测试集
共享卡上用 gpu_memory_utilization 控制占用，默认 0.30(~10G on 32G)。
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward import extract_final_number  # noqa: E402
from train_grpo import SYSTEM_PROMPT  # noqa: E402


def _nums_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return str(a).strip() == str(b).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=0, help="0 = 全测试集 1319")
    ap.add_argument("--gpu_mem", type=float, default=0.30)
    ap.add_argument("--max_tokens", type=int, default=512)
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.n and args.n > 0:
        ds = ds.select(range(min(args.n, len(ds))))
    questions = [ex["question"] for ex in ds]
    answers = [ex["answer"] for ex in ds]
    print(f"[data] 测试题数: {len(questions)}", flush=True)

    llm = LLM(
        model=args.model,
        enable_lora=args.adapter is not None,
        max_lora_rank=16,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=2048,
    )
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    def accuracy(lora_req):
        outs = llm.generate(prompts, sp, lora_request=lora_req)
        correct = 0
        for o, a in zip(outs, answers):
            if _nums_equal(extract_final_number(o.outputs[0].text), extract_final_number(a)):
                correct += 1
        return correct, len(outs)

    cb, nb = accuracy(None)
    print(f"==== BASELINE = {cb/nb:.4f}  ({cb}/{nb}) ====", flush=True)

    if args.adapter:
        lora = LoRARequest("grpo", 1, args.adapter)
        cp, npp = accuracy(lora)
        print(f"==== POST(GRPO) = {cp/npp:.4f}  ({cp}/{npp}) ====", flush=True)

        p1, p2 = cb / nb, cp / npp
        pool = (cb + cp) / (nb + npp)
        se = math.sqrt(pool * (1 - pool) * (1 / nb + 1 / npp))
        z = (p2 - p1) / se if se > 0 else 0.0
        print(f"==== DELTA = +{(p2 - p1) * 100:.2f} pts | z = {z:.2f} | "
              f"{'p<0.05 显著 ✅' if abs(z) >= 1.96 else 'p≥0.05 未显著'} ====", flush=True)


if __name__ == "__main__":
    main()
