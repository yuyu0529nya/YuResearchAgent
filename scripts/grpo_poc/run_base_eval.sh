#!/bin/bash
# base 模型评测(few-shot)。用法: bash run_base_eval.sh <n> [adapter目录]
source "$HOME/grpo_run/venv/bin/activate"
export HF_ENDPOINT=https://hf-mirror.com HF_HOME="$HOME/grpo_run/hf-cache"
export CUDA_VISIBLE_DEVICES=1
cd "$HOME/grpo_run/scripts"
pkill -9 -f eval_gsm8k.py 2>/dev/null
sleep 2
N=${1:-200}
ADAPTER_ARG=""
LOG="$HOME/grpo_run/eval_base_basemodel.log"
if [ -n "$2" ]; then ADAPTER_ARG="--adapter $2"; LOG="$HOME/grpo_run/eval_post_base.log"; fi
nohup python -u eval_gsm8k.py --model Qwen/Qwen2.5-1.5B --base --n "$N" $ADAPTER_ARG \
  < /dev/null > "$LOG" 2>&1 &
echo "base eval launched pid $! (n=$N, log=$LOG)"
