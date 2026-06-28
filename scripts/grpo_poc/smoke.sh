#!/bin/bash
# 冒烟：验证 7B GRPO 管线能跑通一步（meta 修复后）
source /root/miniconda3/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
cd /root/grpo_poc_yu
pkill -9 -f train_grpo.py 2>/dev/null
sleep 3
nohup python -u train_grpo.py --smoke \
  --model /root/autodl-tmp/models/qwen25-7b-instruct \
  --batch_size 4 --num_generations 4 --max_completion_len 256 \
  < /dev/null > /root/grpo_poc_yu/smoke.log 2>&1 &
echo "smoke launched pid $!"
