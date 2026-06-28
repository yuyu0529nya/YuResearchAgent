#!/bin/bash
# 在 ~/grpo_run 下建自包含 venv（可一键 rm -rf ~/grpo_run 删除）
set -e
cd "$HOME/grpo_run"
python3 -m venv venv
source venv/bin/activate
export PIP_CACHE_DIR="$HOME/grpo_run/pip-cache"
python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "=== [1/2] 安装 torch (Blackwell/cu128, 官方源) ==="
pip install torch --index-url https://download.pytorch.org/whl/cu128
echo "=== [2/2] 安装 trl/peft/transformers/datasets/accelerate (清华源) ==="
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "trl>=1.0" peft transformers datasets accelerate
echo "=== 验证 ==="
python -c "import torch,trl,peft,transformers; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'trl',trl.__version__)"
echo "INSTALL_DONE"
