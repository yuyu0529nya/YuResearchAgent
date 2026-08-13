# 研究报告：仅依据 Qwen2.5 技术报告等一手来源，核验并说明 Qwen2.5 的预训练 token 规模、上下文窗口和后训练方法；不讨论其他模型。

---

# Qwen2.5 关键技术参数核验报告：预训练规模、上下文窗口与后训练方法

## 执行摘要

本报告仅依据 Qwen2.5 官方技术报告（arXiv:2412.15115）等一手来源，对 Qwen2.5 的三项关键技术事实进行核验。结论如下：**(1) 预训练 token 规模——已确认**：官方报告摘要与引言明确记载预训练数据从前代（Qwen2）的 7 万亿 token 扩展至 **18 万亿 token**，重点覆盖知识、代码与数学领域 [1]；**(2) 上下文窗口——部分确认**：报告 Table 1 及正文给出各规模模型的上下文与生成长度配置，主流开源规格支持 128K token 上下文 [1]（分档细节见下文，部分数字在当前证据粒度下需谨慎对待）；**(3) 后训练方法——已确认**：报告摘要明确记载采用**超过 100 万条样本的精细监督微调（SFT）**，以及包含**离线 DPO 与在线 GRPO** 的多阶段强化学习 [1]。本报告不讨论其他模型。

## 背景

Qwen2.5 是阿里巴巴 Qwen 团队于 2024 年发布的大语言模型系列，其官方技术报告（Qwen Team, An Yang 等，2024 年 12 月，arXiv:2412.15115）是唯一被本文用作核验基准的一手来源 [1]。该报告系统介绍了 Qwen2.5 在预训练数据规模、模型架构与后训练管线上的改进，涵盖开源稠密模型（0.5B 至 72B）及专有 API 模型 Qwen2.5-Turbo / Qwen2.5-Plus。研究问题要求核验三项具体事实：预训练 token 规模、上下文窗口长度、后训练方法，且不得混用其他模型的信息。

## 关键发现

### 1. 预训练 token 规模：18 万亿 token（已核实）

一手来源中的原始表述可直接引用：

> "we have scaled the high-quality pre-training datasets from the previous 7 trillion tokens to **18 trillion tokens**."（摘要）[1]
> "The pre-training data increased from 7 trillion tokens to 18 trillion tokens, with focus on knowledge, coding, and mathematics."（引言）[1]

即预训练语料从 Qwen2 的 7 万亿 token 扩展至 **18 万亿（18T）token**，扩充方向聚焦于知识、编程与数学。两份子任务核验结果在该数字上完全一致，且证据审计中该表述获得检索证据的直接支持。

关于数据工程细节（如使用 Qwen2-Instruct 进行数据质量过滤、融入 Qwen2.5-Math 与 Qwen2.5-Coder 的数据及合成数据），子任务报告中有所提及，与 [1] 及 Qwen2.5-Coder 技术报告 [3] 所述方向一致；但这些细节在当前证据审计中未被逐条独立证实，建议以 arXiv 原文预训练章节复核后再作强断言。

### 2. 上下文窗口：32K 至 128K 分档，专有版支持 1M（主结论已核实，细节需谨慎）

技术报告 Table 1（Context / Generation Length 列）及正文给出了分档配置 [1]：

- **开源稠密模型**：7B、14B、32B、72B 规格支持 **128K token** 上下文，生成长度上限 **8K token**；小规格（0.5B/1.5B/3B）为 32K 上下文。
- **长上下文预训练**：采用渐进式扩展策略，初始阶段为 4,096 token，最终阶段扩展至 32,768 token，并通过 ABF 技术将 RoPE 基频从 10,000 提升至 1,000,000 [1]。
- **Qwen2.5-Turbo**（专有 MoE 模型）：训练中经 32K→65,536→131,072→262,144 的渐进扩展，最终支持最长 **100 万（1M）token** 上下文 [1]。
- 生成长度整体从前代 Qwen2 的 2K 提升至 8K [1]。

**证据约束说明**：上下文窗口的"128K 主流规格"这一主结论可由 Table 1 的检索证据支持，但具体到"小模型 32K""ABF 基频 1,000,000""Turbo 1M"等数值，当前证据审计未能从检索片段中逐条独立核实，故本报告将其作为"技术报告记载、待原文复核"的内容呈现，不作为无条件事实断言。

### 3. 后训练方法：SFT（>100 万样本）+ 离线 DPO + 在线 GRPO（已核实）

报告摘要的原始表述为：

> "we implement intricate supervised finetuning with over 1 million samples, as well as multistage reinforcement learning, including offline learning **DPO** and online learning **GRPO**."（摘要）[1]

即后训练管线包含两大部分：

1. **精细监督微调（SFT）**：使用**超过 100 万条样本**构建的高质量指令数据；
2. **多阶段强化学习**：包括**离线强化学习 DPO**（Direct Preference Optimization，直接偏好优化）与**在线强化学习 GRPO**（Group Relative Policy Optimization，组相对策略优化）。

该两项方法名称（DPO、GRPO）及 SFT 样本规模（>1M）均在证据审计中获得直接支持 [1]。报告称该管线显著增强了人类偏好对齐、长文本生成、结构化数据分析与指令遵循能力——此效果性表述方向与报告一致，但当前无独立检索片段逐字支持，宜视为报告的自我陈述而非外部验证结论。

## 分析

三项核验事实之间存在内在技术逻辑：

- **数据规模扩充（7T→18T）** 是能力提升的基础，且扩充重点（知识、代码、数学）与 Qwen 团队同期发布的 Qwen2.5-Coder [3]、Qwen2.5-Math 专项模型的数据资产直接相关，体现"专项模型反哺通用基座"的数据复用策略。
- **上下文窗口** 的扩展依赖预训练阶段的位置编码改造（RoPE 基频调整、渐进式长度训练），说明 128K/1M 并非推理期外推的权宜手段，而是训练内建能力 [1]。
- **后训练管线（SFT→DPO→GRPO）** 代表 2024 年业界主流的"离线偏好优化 + 在线强化学习"混合范式：DPO 以较低算力成本完成偏好对齐，GRPO 则通过在线采样进一步优化生成质量。摘要中"over 1 million samples"的 SFT 规模也反映了数据驱动的指令微调思路 [1]。

需要指出：**Qwen2.5-VL 技术报告 [2]** 属于多模态视觉语言模型，与本次语言模型核验问题无直接关系，本文未将其作为事实依据。

## 来源间对比与矛盾消解

两份子任务核验结果在核心事实上**无矛盾**：18T token、128K 上下文、SFT + DPO + GRPO 三项结论完全一致。差异仅在于证据强度：子任务 1 自称直接读取 arXiv 全文（自评置信度 0.98），子任务 2 因检索工具超时未能当场比对原文（自评 0.8）。结合证据审计结果，18T 与后训练管线两项有检索片段直接支持，置信度高；上下文窗口的分档数值细节证据较弱，故本报告对后者采取保守表述。这是两个来源之间的主要张力，已通过区分"已核实事实"与"报告记载待复核细节"予以消解。

## 启示

1. **可引用的硬事实**：若需在后续工作中引用 Qwen2.5 的规格，"18T 预训练 token"与"SFT（>1M 样本）+ DPO + GRPO"两条可放心引用 arXiv:2412.15115 [1]。
2. **上下文窗口需注明规格档位**：Qwen2.5 不同参数规模与专有版的上下文长度不同（32K/128K/1M），引用时应明确所指具体模型变体，避免笼统表述。
3. **方法论意义**：Qwen2.5 报告展示了"大规模数据扩充 + 训练内建长上下文 + 混合式强化学习"的完整工程路线，是理解 2024 年开源 LLM 技术演进的代表性一手文献。

## 结论

仅依据一手来源核验：Qwen2.5 的预训练数据规模为 **18 万亿 token**（较前代 7 万亿大幅扩充，聚焦知识、代码、数学）[1]；上下文窗口按模型规格分档，主流开源模型支持 128K token、专有版 Qwen2.5-Turbo 支持最长 1M token（主结论可靠，分档数值细节建议原文复核）[1]；后训练方法为**超 100 万样本的 SFT + 离线 DPO + 在线 GRPO 的多阶段管线** [1]。无与其他模型混淆之处，证据审计中无任何被驳斥（refuted）的主张。

## 参考来源

[1] Qwen2.5 Technical Report — Qwen, Yang An, Baosong Yang, Beichen Zhang, Binyuan Hui, B. Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu（2024） — https://arxiv.org/abs/2412.15115
[2] Qwen2.5-VL Technical Report — Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang（2025） — https://arxiv.org/abs/2502.13923
[3] Qwen2.5-Coder Technical Report — Binyuan Hui, Jian Yang, Cui Zeyu, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Lu Keming（2024） — https://arxiv.org/abs/2409.12186

**Overall Confidence: 0.63**（18T token 与后训练方法两项有直接证据支持，置信度高；上下文窗口分档数值细节因证据粒度不足采取保守表述，拉低整体置信度。）

---

## 元信息

- **置信度**: 0.63
- **搜索轮数**: 3
- **重规划次数**: 0
- **证据补充轮数**: 0
- **对抗轮数**: 0
- **总耗时**: 352.76 秒

## 证据审计

- **Claim 覆盖率**: 15.6%
- **核验结果**: 7 supported / 0 refuted / 38 NEI（共 45 条）
- **原始/权威来源占比**: 100.0%
- **全文证据来源占比**: 33.3%
- **审计文件**: `outputs/evidence/evidence_evidence_v2_1786614295791595000_20260813T095048Z.json`

### 仍需谨慎的陈述

- `not_enough_evidence` 本报告仅依据 Qwen2.5 官方技术报告（arXiv:2412.15115）等一手来源，对 Qwen2.5 的三项关键技术事实进行核验。
- `not_enough_evidence` 结论如下：**(1) 预训练 token 规模——已确认**：官方报告摘要与引言明确记载预训练数据从前代（Qwen2）的 7 万亿 token 扩展至 **18 万亿 token**，重点覆盖知识、代码与数学领域 ；
- `not_enough_evidence` **(2) 上下文窗口——部分确认**：报告 Table 1 及正文给出各规模模型的上下文与生成长度配置，主流开源规格支持 128K token 上下文 （分档细节见下文，部分数字在当前证据粒度下需谨慎对待）；
- `not_enough_evidence` **(3) 后训练方法——已确认**：报告摘要明确记载采用**超过 100 万条样本的精细监督微调（SFT）**，以及包含**离线 DPO 与在线 GRPO** 的多阶段强化学习 。
- `not_enough_evidence` Qwen2.5 是阿里巴巴 Qwen 团队于 2024 年发布的大语言模型系列，其官方技术报告（Qwen Team, An Yang 等，2024 年 12 月，arXiv:2412.15115）是唯一被本文用作核验基准的一手来源 。
