#!/bin/bash
# 1.5B GRPO 训练 — 锁 GPU 1，保守配置(共享卡 ~12.8G 空闲，不挤爆同事)
source "$HOME/grpo_run/venv/bin/activate"
export HF_ENDPOINT=https://hf-mirror.com HF_HOME="$HOME/grpo_run/hf-cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
cd "$HOME/grpo_run/scripts"
pkill -9 -f train_grpo.py 2>/dev/null
sleep 3
nohup python -u train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir "$HOME/grpo_run/outputs/grpo_qwen1.5b_gsm8k" \
  --batch_size 8 --num_generations 8 --max_completion_len 320 \
  --max_steps 400 \
  < /dev/null > "$HOME/grpo_run/train.log" 2>&1 &
echo "train pid $!"
