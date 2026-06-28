#!/bin/bash
# 基线评测(原始 1.5B)— 锁 GPU 1
source "$HOME/grpo_run/venv/bin/activate"
export HF_ENDPOINT=https://hf-mirror.com HF_HOME="$HOME/grpo_run/hf-cache"
export CUDA_VISIBLE_DEVICES=1
cd "$HOME/grpo_run/scripts"
pkill -9 -f eval_gsm8k.py 2>/dev/null
sleep 2
nohup python -u eval_gsm8k.py \
  --model Qwen/Qwen2.5-1.5B-Instruct --n 200 \
  < /dev/null > "$HOME/grpo_run/eval_base.log" 2>&1 &
echo "baseline eval pid $!"
