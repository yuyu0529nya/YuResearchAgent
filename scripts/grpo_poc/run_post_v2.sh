#!/bin/bash
# v2 训练后评测(加载改进版 LoRA adapter, n=500 与基线同口径)— 锁 GPU 1
source "$HOME/grpo_run/venv/bin/activate"
export HF_ENDPOINT=https://hf-mirror.com HF_HOME="$HOME/grpo_run/hf-cache"
export CUDA_VISIBLE_DEVICES=1
cd "$HOME/grpo_run/scripts"
pkill -9 -f eval_gsm8k.py 2>/dev/null
sleep 2
nohup python -u eval_gsm8k.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter "$HOME/grpo_run/outputs/grpo_qwen1.5b_v2" --n 500 \
  < /dev/null > "$HOME/grpo_run/eval_post_v2.log" 2>&1 &
echo "post v2 eval pid $!"
