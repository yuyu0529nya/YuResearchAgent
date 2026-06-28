# 1.5B GRPO 小模型 runbook（12G+ 单卡）

目标：用 Qwen2.5-1.5B-Instruct 在 GSM8K 上跑 GRPO，拿一个**涨幅明显**的 X%→Y%
（1.5B 基线低、有空间，比 7B 近天花板的 +2 好看得多）。

## 0. 环境（新租机 / 自己的 3060 通用）
```bash
# 5090(Blackwell) 用 cu128；3060/4090(Ampere/Ada) 用 cu121 或 cu124：
#   3060/4090:  pip install torch --index-url https://download.pytorch.org/whl/cu124
#   5090:       pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -U "trl>=1.0" peft transformers datasets accelerate   # 若 base 已有大半，可加 --no-deps 单装 trl
python -c "import torch,trl,peft; print(torch.cuda.get_device_name(0), 'trl', trl.__version__)"
```
> 把本目录的 `reward.py`、`train_grpo.py`、`eval_gsm8k.py`、`train_1.5b.sh` 传到机器上同一目录。

## 1. 基线评测（训练前）
```bash
export HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0
python -u eval_gsm8k.py --model Qwen/Qwen2.5-1.5B-Instruct --n 200 | tee eval_base_1.5b.log
# 记下 baseline 准确率（预计 ~0.55-0.62）
```

## 2. GRPO 训练（约 400 步）
```bash
bash train_1.5b.sh          # 后台跑，日志 train_1.5b.log
tail -f train_1.5b.log      # 看 reward / 步数；1.5B 比 7B 快很多
```
- 12G OOM → 编辑 train_1.5b.sh：`--batch_size 4` 或 `--max_completion_len 256`
- 24G+ 想吃满 → `--batch_size 16 --max_completion_len 512`

## 3. 训练后评测
```bash
python -u eval_gsm8k.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/grpo_qwen1.5b_gsm8k --n 200 | tee eval_post_1.5b.log
# 对比 baseline，差值即 GRPO 增益（1.5B 预期能看到清晰的几到十几个点）
```

## 4. 关机前拉证据回本地
需要 train_1.5b.log + eval_base_1.5b.log + eval_post_1.5b.log + outputs/grpo_qwen1.5b_gsm8k/(adapter)。
（我可以像 7B 那次一样用 SSH 帮你拉回 scripts/grpo_poc/results_1.5b/。）

## 诚实预期
- 1.5B 基线 ~55-60%，GRPO 后**有望清晰提升**（不像 7B 在噪声内）。
- 若提升不明显：加大 `--max_steps`（如 600）、确认 reward 在升；GSM8K 上 1.5B 通常对 GRPO 有响应。
