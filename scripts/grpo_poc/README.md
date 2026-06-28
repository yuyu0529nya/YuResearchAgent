# GRPO PoC — 单卡 (RTX 5090) 真训练实证

在一张 RTX 5090 (32GB) 上，用 **TRL GRPOTrainer** 对 **Qwen2.5-1.5B-Instruct** 做
**LoRA + GRPO** 训练，任务为 **GSM8K**（数学，可验证奖励 = 答案对错），
产出"**准确率随训练上升**"的真实曲线。

这是 YuResearchAgent 中 M6 自进化引擎里"GRPO 梯度更新"那一步的**独立、最小、可运行**实证：
M6 负责出题 / 收集轨迹 / 塑形 reward / 导出 VERL 数据，本目录验证其指向的 **GRPO trainer 真能端到端跑通并提分**。

> 注：本 PoC 刻意**不在多轮研究 agent 上训练**——那样每个 rollout 是一次完整研究（分钟级），
> GRPO 每轮上百次 rollout 不现实。用 GSM8K 单轮可验证任务，是证明"训练环路成立"的高性价比选择。

---

## 0. 租机要求
- **GPU**：RTX 5090 (32GB) ×1（1.5B + LoRA 很宽裕；同代码可上 3B）
- **系统**：Linux + **CUDA 12.8**（Blackwell sm_120 必需）。RunPod / Vast.ai 选 "CUDA 12.8" 镜像
- 磁盘 ≥ 30GB（模型 + 数据集缓存）

## 1. 安装
```bash
python -m venv .venv && source .venv/bin/activate
# Blackwell 必须 cu128 轮子：
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# 验证 GPU 可见：
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
# 期望：NVIDIA GeForce RTX 5090 True
```

## 2. 冒烟测试（先跑这个！几分钟，只验证管线不报错）
```bash
python train_grpo.py --smoke
```
看到 reward 日志在滚动、无报错、`outputs/` 下出现 checkpoint，即管线通。**确认通过再开正式训练，省钱。**

## 3. 评测基线（训练前准确率）
```bash
python eval_gsm8k.py --model Qwen/Qwen2.5-1.5B-Instruct --n 200
# 记下 baseline 准确率，例如 ~0.55
```

## 4. 正式 GRPO 训练
```bash
python train_grpo.py --max_steps 300 --output_dir outputs/grpo_qwen1.5b
# 关注日志：rewards/correctness_reward 应随 step 上升
# 1.5B / 300 步 / 单 5090：约 1–3 小时（不开 vLLM）
```
> 提速可加 `--use_vllm`（需 vLLM 支持 Blackwell；不稳就别开）。

## 5. 评测训练后（加载 LoRA adapter）
```bash
python eval_gsm8k.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/grpo_qwen1.5b --n 200
# 对比 baseline，差值即 GRPO 真实增益
```

## 6. 把结果发我
把这几样贴回来，我帮你写成简历线 + 排查异常：
- 训练日志里 `reward` / `correctness_reward` 随 step 的变化（几行即可）
- 第 3 步 baseline 准确率 与 第 5 步训练后准确率

---

## 预期 & 诚实边界
- **预期**：1.5B 在 GSM8K 上 GRPO 后准确率有可测上升（典型几个百分点到十几个百分点，取决于步数/超参）。
- **能写进简历**：`端到端跑通 GRPO 训练（TRL + Qwen2.5-1.5B + LoRA，单卡），GSM8K 准确率 X%→Y%`。
- **不能写**：把这个 PoC 的提分套到主研究 agent 上——它们是两件事，PoC 验证的是"训练环路与梯度更新成立"。
- **若 reward 不升**：多半是 LoRA 学习率 / 步数 / num_generations 需调；先把 `--max_steps` 调大、`--num_generations 8` 保持，或换 `--model Qwen/Qwen2.5-3B-Instruct`。

## 故障速查
| 现象 | 处理 |
|---|---|
| `torch.cuda.is_available()` 为 False | 装错轮子；务必用 cu128 index 重装 torch |
| OOM | 降 `--batch_size`、`--max_completion_len`，或 `--num_generations 4` |
| vLLM 装不上/崩 | 去掉 `--use_vllm`，用 HF generate |
| GRPOTrainer 参数报错 | TRL API 随版本有变；`pip show trl` 看版本，按该版文档微调 `GRPOConfig` 字段 |
