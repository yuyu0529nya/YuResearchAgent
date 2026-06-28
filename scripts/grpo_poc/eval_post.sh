#!/bin/bash
# 训练后评测：加载 GRPO 训练的 LoRA adapter，对比基线
source /root/miniconda3/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf-cache
export CUDA_VISIBLE_DEVICES=0
cd /root/grpo_poc_yu
pkill -9 -f eval_gsm8k.py 2>/dev/null
sleep 2
nohup python -u eval_gsm8k.py \
  --model /root/autodl-tmp/models/qwen25-7b-instruct \
  --adapter outputs/grpo_qwen7b_gsm8k \
  --n 100 \
  < /dev/null > /root/grpo_poc_yu/eval_post.log 2>&1 &
echo "post-train eval launched pid $!"
