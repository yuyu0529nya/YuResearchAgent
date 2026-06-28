#!/bin/bash
# base 模型 GRPO（R1-Zero 式：few-shot 冷启动 + 格式&正确率奖励）— 锁 GPU 1
source "$HOME/grpo_run/venv/bin/activate"
export HF_ENDPOINT=https://hf-mirror.com HF_HOME="$HOME/grpo_run/hf-cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
cd "$HOME/grpo_run/scripts"
pkill -9 -f train_grpo.py 2>/dev/null
sleep 3
nohup python -u train_grpo.py \
  --base --model Qwen/Qwen2.5-1.5B \
  --output_dir "$HOME/grpo_run/outputs/grpo_base" \
  --batch_size 8 --num_generations 8 --max_completion_len 320 \
  --grad_accum 4 --temperature 1.0 \
  --max_steps 500 \
  < /dev/null > "$HOME/grpo_run/train_base.log" 2>&1 &
echo "base train launched pid $!"
