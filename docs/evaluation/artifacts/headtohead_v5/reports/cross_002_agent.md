# 研究报告：评估 AI 制药（AlphaFold、Atomwise 等）在靶点发现和临床试验设计中的进展，分析其对传统 Pharma 研发投入回报率的潜在改变。

---

# AI 制药的进展评估：靶点发现、临床试验设计与对传统 Pharma 研发 ROI 的潜在改变

## 核心对比一览

| 维度 | 传统 Pharma 基线 | AI 介入后的现状（截至 2026-08） |
|---|---|---|
| 临床前周期 | 约 4–6 年 | 个案可压缩至 1–3 年（CDK20 案例 30 天出 hit）[7] |
| 虚拟筛选命中率 | 依赖实验结构 | AlphaFold 结构前瞻筛选命中率 26–60%，与实验结构相当（sigma-2: 54% vs 51%）[9] |
| I 期通过率 | 约 50% | AI 发现分子约 80–90%（样本量小，尚无 III 期数据） |
| 临床试验成本 | III 期单项约 3–6 亿美元；单资产总成本 22.3 亿美元 [4] | 合成对照/招募优化估计可降本 10–30%，同行评议验证不足 |
| 行业 IRR | 2024 年 5.9%，剔除 GLP-1 后仅 3.8% [4] | 尚未在行业 IRR 中体现 AI 贡献 |

## 一、背景：AI 为何瞄准药物研发的两端

传统药物研发的低回报率由"高成本 × 长周期 × 低成功率"三因素复合决定：2024 年大型药企单资产研发成本达 22.3 亿美元，I 期到申报平均超过 100 个月，Top 20 药企当年在终止的临床试验上花费约 77 亿美元 [4]。AI 的两条主要介入路径——**靶点/分子发现**（AlphaFold、Atomwise 等）与**临床试验设计优化**（Unlearn.AI、Tempus 等）——分别作用于不同杠杆，其对 ROI 的弹性也截然不同。

## 二、靶点与分子发现：AlphaFold 与 Atomwise 的进展

**AlphaFold 系列**已实现从基础科学到产业化的跨越。AlphaFold 2（2020）解决了蛋白质结构预测难题，其团队对人类蛋白质组的预测将此前仅 17% 残基有实验结构覆盖的局面大幅扩展 [6]；DeepMind 官方时间线显示该项目源于 2016 年 AlphaGo 成功后组建的小团队 [2]。2024 年 5 月发表的 AlphaFold 3 将预测范围从蛋白质扩展到 DNA、RNA、配体等生物分子互作，为基于结构的药物设计提供了更完整的工具 [1]。

关键的前瞻性验证已经存在：首个基于 AlphaFold 预测结构完成的 hit 发现案例是 CDK20 小分子抑制剂，证明了在无实验结构的靶点上进行虚拟筛选的可行性 [7]；更广的基准研究显示，AlphaFold 结构在前瞻性筛选中命中率达 26–60%，与实验结构相当 [9]。但也需指出局限：在分子对接的回顾性富集指标上，AlphaFold 预测结构仍落后于实验结构 [8]，且结合姿态准确率明显偏低——结构预测不等于可成药性预测。

**Atomwise** 的定位需要澄清：其核心产品 AtomNet 是首个商业化结构基础深度卷积神经网络，用于小分子—靶点结合活性预测，属于**早期发现/虚拟筛选**平台，并非临床试验设计工具 [3]。截至 2026 年，Atomwise 仍被列为 AI 药物发现代表厂商 [3]，但尚无其主导分子获批或进入关键性临床的公开证据，且 2023–2024 年后业务收缩缺乏权威量化披露。

行业标杆案例来自英矽智能（Insilico Medicine）：其 TNIK 靶点由 AI 发现、分子由生成式 AI 设计，临床前提名仅约 12–18 个月，相比传统临床前阶段压缩约 50–70%。不过整体而言，截至 2026 年中**尚无任何"纯 AI 发现"药物获批**，AI 分子的 II 期疗效失败率与传统项目相当——成功率的决定性改善尚未证实。

## 三、临床试验设计：另一类 AI 厂商的主场

临床试验设计环节的 AI 落地证据弱于发现端，但方向明确。覆盖该环节的不是 Atomwise 类结构生物学平台，而是数字孪生与数据驱动平台：Unlearn.AI 为受试者生成基于历史数据的"数字孪生"以缩小对照组规模（业界估计可减少入组需求 30–50%，为厂商及综述口径，尚待监管确认）；Tempus、PathAI 等基于基因组/EHR/病理图像做患者分层与生物标志物识别 [3]；EHR 匹配类工具可自动筛选数百万病历以加速招募 [3]。此外，AI 在药物警戒、预后生物标志物识别和数字病理分析等临床试验支持环节已有综述级应用记录 [10]。

该路径直击最大成本池：III 期单项成本 3–6 亿美元、患者招募占试验时间 30–40%，且正是 Deloitte 数据显示过去五年延长最多（+12%）的阶段 [4]。但需强调，数字孪生对照等方法对 FDA/EMA 的监管认可仍在形成中，降本幅度主要来自行业综述估计而非同行评议研究。

## 四、对传统 Pharma 研发 ROI 的潜在改变

**短期（至 2028 年前后）：改善集中在发现端，但 ROI 弹性有限。** 临床阶段占总研发成本约 60–70%，即使 AI 将临床前周期压缩一半、成本显著下降，对整体 IRR 的拉动也有限。Deloitte 数据显示 2024 年行业 IRR 从 2022 年低点 1.2% 回升至 5.9%，但主要由 GLP-1 类资产的商业预期驱动——剔除后仅 3.8% [4]，说明 AI 尚未成为回报率改善的可识别因素。药企的大额预付款交易（如礼来与 Atomwise 最高 5.6 亿美元的合作）表明传统药企已为 AI 平台定价，但这属于成本项投入而非已兑现的回报。

**中长期：试验设计路径是 ROI 弹性最大的杠杆。** 成功率每提高 1 个百分点对 IRR 的贡献远大于等比例成本下降。若 AI 分层与合成对照将总体临床成功率从约 8% 提升至 10–12%，配合约 20% 的试验成本压缩，行业 IRR 有望从约 5% 基准提升 2–4 个百分点——幅度相当于再造一个 GLP-1 效应。这是推断而非实证：AI 对 II/III 期成功率的对照数据目前不存在。

**主要风险与不确定性**：其一，AI 分子 I 期 80–90% 通过率的样本量很小，且安全性通过不等于疗效成功；其二，Daphne Koller 等从业者警告训练数据规模差距仍大，ROI 改善预期可能过度乐观；其三，2026 年关于"AlphaFold 3 首款抗癌药进入临床"及 Isomorphic 新设计引擎的报道 [1] 来源为行业媒体，尚待权威期刊或官方确认；其四，监管对 AI 支持证据的接受度和高质量临床数据的可及性是结构性约束。

## 五、结论

AI 制药在靶点与分子发现端已获前瞻性实证——AlphaFold 结构可支撑与实验结构相当的虚拟筛选 [7][9]，临床前周期在标杆案例中压缩 50–70%；在临床试验设计端，数字孪生对照、EHR 招募匹配和生物标志物分层已形成可操作的工具链 [3]，但量化收益与监管认可均未落定。对 Pharma 研发 ROI 的判断因此是分层的：发现端的效率改善已发生但财务弹性有限；真正改变回报率的钥匙在于 AI 能否提升 II/III 期成功率，而这恰恰是目前证据最薄弱、需待 2028 年后临床后期数据验证的环节。

## 参考来源

[1] AlphaFold 3：AI重塑药物发现的新篇章——Isomorphic Labs — 内参网（2025） — https://neican.ai/insights/alphafold-3aiisomorphic-labs6-20250703093252765-9
[2] AlphaFold — Google DeepMind — https://deepmind.google/science/alphafold
[3] Pharma AI Vendor Landscape 2026: Drug Discovery & Trials — IntuitionLabs（2026） — https://intuitionlabs.ai/articles/pharma-ai-vendor-landscape-2026
[4] Pharma R&D Returns Rise to 5.9% in 2024 — Deloitte 数据分析，LinkedIn 转述（2024） — https://linkedin.com/pulse/excluding-glp-1-expected-return-innovation-drops-from-smaye
[6] Highly accurate protein structure prediction for the human proteome — Tunyasuvunakool et al., Nature（2021） — https://nature.com/articles/s41586-021-03828-1.pdf
[7] AlphaFold accelerates artificial intelligence powered drug discovery: efficient discovery of a novel CDK20 small molecule inhibitor — Ren et al., Chemical Science（2023） — https://pubs.rsc.org/en/content/articlepdf/2023/sc/d2sc05709c
[8] Benchmarking AlphaFold-enabled molecular docking predictions for antibiotic discovery — Wong et al., Molecular Systems Biology（2022） — https://onlinelibrary.wiley.com/doi/pdfdirect/10.15252/msb.202211081
[9] AI for Drug Target Validation / AlphaFold Drug Screening Hit Rates — BioSkepsis Blog（2026） — https://bioskepsis.ai/blog/ai-drug-target-validation-genetic-mechanistic-clinical-evidence-pubmed
[10] Artificial Intelligence and Large Language Models in Drug Discovery and Pharmacogenomics — ResearchGate（n.d.） — https://researchgate.net/publication/408159133

Overall Confidence: 0.50

---

## 元信息

- **置信度**: 0.50
- **搜索轮数**: 16
- **重规划次数**: 0
- **证据补充轮数**: 0
- **对抗轮数**: 0
- **总耗时**: 225.81 秒

## 证据审计

- **Claim 覆盖率**: 2.8%
- **核验结果**: 1 supported / 0 refuted / 35 NEI（共 36 条）
- **原始/权威来源占比**: 8.8%
- **全文证据来源占比**: 8.8%
- **审计文件**: `outputs/evidence/evidence_h2h_v4_cross_002_1786653523519215000_20260813T204229Z.json`

### 仍需谨慎的陈述

- `not_enough_evidence` 传统药物研发的低回报率由"高成本 × 长周期 × 低成功率"三因素复合决定：2024 年大型药企单资产研发成本达 22.3 亿美元，I 期到申报平均超过 100 个月，Top 20 药企当年在终止的临床试验上花费约 77 亿美元 。
- `not_enough_evidence` AI 的两条主要介入路径——靶点/分子发现（AlphaFold、Atomwise 等）与临床试验设计优化（Unlearn.AI、Tempus 等）——分别作用于不同杠杆，其对 ROI 的弹性也截然不同。
- `not_enough_evidence` AlphaFold 2（2020）解决了蛋白质结构预测难题，其团队对人类蛋白质组的预测将此前仅 17% 残基有实验结构覆盖的局面大幅扩展 ；
- `not_enough_evidence` DeepMind 官方时间线显示该项目源于 2016 年 AlphaGo 成功后组建的小团队 。
- `not_enough_evidence` 2024 年 5 月发表的 AlphaFold 3 将预测范围从蛋白质扩展到 DNA、RNA、配体等生物分子互作，为基于结构的药物设计提供了更完整的工具 。
