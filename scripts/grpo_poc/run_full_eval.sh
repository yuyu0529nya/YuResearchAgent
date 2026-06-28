#!/bin/bash
# 大样本评测(HF generate)：基线+训练后顺序跑，脚本内部自后台化。用法: bash run_full_eval.sh <n>
N=${1:-500}
cd "$HOME/grpo_run/scripts"
pkill -9 -f eval_gsm8k.py 2>/dev/null
pkill -9 -f vllm_eval.py 2>/dev/null
sleep 3
nohup bash -c "
  source \"\$HOME/grpo_run/venv/bin/activate\"
  export HF_ENDPOINT=https://hf-mirror.com HF_HOME=\"\$HOME/grpo_run/hf-cache\" CUDA_VISIBLE_DEVICES=1
  cd \"\$HOME/grpo_run/scripts\"
  echo '=== baseline n=$N ==='
  python -u eval_gsm8k.py --model Qwen/Qwen2.5-1.5B-Instruct --n $N > \"\$HOME/grpo_run/eval_base_full.log\" 2>&1
  echo '=== post n=$N ==='
  python -u eval_gsm8k.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter \"\$HOME/grpo_run/outputs/grpo_qwen1.5b_gsm8k\" --n $N > \"\$HOME/grpo_run/eval_post_full.log\" 2>&1
  echo 'BOTH_DONE'
" < /dev/null > "$HOME/grpo_run/eval_full_driver.log" 2>&1 &
echo "full eval launched pid $! (n=$N)"
