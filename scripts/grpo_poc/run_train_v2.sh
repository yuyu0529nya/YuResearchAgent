#!/bin/bash
# 1.5B GRPO 改进版：梯度累积4(每更新4道不同题) + 温度1.0(减少零优势) + 500步(~2000题)
source "$HOME/grpo_run/venv/bin/activate"
export HF_ENDPOINT=https://hf-mirror.com HF_HOME="$HOME/grpo_run/hf-cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
cd "$HOME/grpo_run/scripts"
pkill -9 -f train_grpo.py 2>/dev/null
sleep 3
nohup python -u train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir "$HOME/grpo_run/outputs/grpo_qwen1.5b_v2" \
  --batch_size 8 --num_generations 8 --max_completion_len 320 \
  --grad_accum 4 --temperature 1.0 \
  --max_steps 500 \
  < /dev/null > "$HOME/grpo_run/train_v2.log" 2>&1 &
echo "train v2 launched pid $!"
