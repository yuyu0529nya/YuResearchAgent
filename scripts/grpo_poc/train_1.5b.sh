#!/bin/bash
# 1.5B GRPO 训练 —— 适配 12G+ 单卡。基线低、跑更多步，涨幅更明显。
# 新机会自动从 HF 镜像下载 1.5B（~3GB）；若本地已有模型，把 MODEL 改成本地路径。
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

source /root/miniconda3/bin/activate 2>/dev/null || source ~/miniconda3/bin/activate 2>/dev/null || true
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$(dirname "$0")"

pkill -9 -f train_grpo.py 2>/dev/null
sleep 3
nohup python -u train_grpo.py \
  --model "$MODEL" \
  --output_dir outputs/grpo_qwen1.5b_gsm8k \
  --batch_size 8 --num_generations 8 --max_completion_len 384 \
  --max_steps 400 \
  < /dev/null > train_1.5b.log 2>&1 &
echo "1.5B train launched pid $!  (log: train_1.5b.log)"
echo "12G 显存若 OOM：把 --batch_size 改 4 或 --max_completion_len 改 256"
echo "卡更大(24G+)想吃满：--batch_size 16 --max_completion_len 512"
