#!/bin/bash
# 正式 GRPO 训练：7B + LoRA，吃满 32G 的大配置
source /root/miniconda3/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
cd /root/grpo_poc_yu
pkill -9 -f train_grpo.py 2>/dev/null
sleep 3
nohup python -u train_grpo.py \
  --model /root/autodl-tmp/models/qwen25-7b-instruct \
  --output_dir outputs/grpo_qwen7b_gsm8k \
  --batch_size 8 --num_generations 8 --max_completion_len 512 \
  --max_steps 300 \
  < /dev/null > /root/grpo_poc_yu/train.log 2>&1 &
echo "train launched pid $!"
