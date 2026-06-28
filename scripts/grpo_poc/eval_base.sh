#!/bin/bash
# 基线评测：原始 7B 在 GSM8K 上的准确率（训练前）
source /root/miniconda3/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf-cache
export CUDA_VISIBLE_DEVICES=0
cd /root/grpo_poc_yu
pkill -9 -f eval_gsm8k.py 2>/dev/null
sleep 2
nohup python -u eval_gsm8k.py \
  --model /root/autodl-tmp/models/qwen25-7b-instruct \
  --n 100 \
  < /dev/null > /root/grpo_poc_yu/eval_base.log 2>&1 &
echo "baseline eval launched pid $!"
