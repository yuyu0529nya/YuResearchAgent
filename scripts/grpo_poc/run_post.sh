#!/bin/bash
# 训练后评测(加载 LoRA adapter)— 锁 GPU 1
source "$HOME/grpo_run/venv/bin/activate"
export HF_ENDPOINT=https://hf-mirror.com HF_HOME="$HOME/grpo_run/hf-cache"
export CUDA_VISIBLE_DEVICES=1
cd "$HOME/grpo_run/scripts"
pkill -9 -f eval_gsm8k.py 2>/dev/null
sleep 2
nohup python -u eval_gsm8k.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter "$HOME/grpo_run/outputs/grpo_qwen1.5b_gsm8k" --n 200 \
  < /dev/null > "$HOME/grpo_run/eval_post.log" 2>&1 &
echo "post-train eval pid $!"
