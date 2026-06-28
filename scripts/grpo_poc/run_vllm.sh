#!/bin/bash
# vLLM 批量评测 baseline+post。用法: bash run_vllm.sh <n>  (n=0 全测试集1319, n=20 冒烟)
source "$HOME/grpo_run/venv/bin/activate"
export HF_ENDPOINT=https://hf-mirror.com HF_HOME="$HOME/grpo_run/hf-cache"
export CUDA_VISIBLE_DEVICES=1
cd "$HOME/grpo_run/scripts"
pkill -9 -f vllm_eval.py 2>/dev/null
sleep 3
N=${1:-0}
nohup python -u vllm_eval.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter "$HOME/grpo_run/outputs/grpo_qwen1.5b_gsm8k" \
  --n "$N" --gpu_mem 0.30 \
  < /dev/null > "$HOME/grpo_run/vllm_eval.log" 2>&1 &
echo "vllm eval launched pid $! (n=$N)"
