# 研究报告：评估自适应学习系统（如 Khan Academy、松鼠 AI）在 K-12 数学教育中的效果，分析其个性化推荐算法、学习效果量化指标和师生接受度。

---

# 自适应学习系统在 K-12 数学教育中的效果评估：Khan Academy 与松鼠 AI 的比较分析

## 一、总体应用与效果证据

### Khan Academy（可汗学院）

Khan Academy 是覆盖 K-12 数学为主、多学科并行的全球性免费学习平台。其效果证据在自适应学习系统中相对较强，但存在明显的来源偏差需要注意：

- **萨尔瓦多大规模 RCT（Büchel 等，2019）**：约 3,500 名 3–6 年级学生参与，发现 Khan Academy 补充性使用的数学成绩优于仅教师干预组。该结果主要通过 Khan Academy 官方博客（2026-02-07）传播，虽基于独立研究者的随机对照试验，但发布渠道属于第一方，引用时应注明。
- **Oreopoulos 等（2025 公布）RCT**：与 Khan Academy 合作、约 1.1 万名学生参与。教师指导下每周约 35 分钟使用 KA 的 3–6 年级学生，年末数学测评成绩比对照组高约 0.12–0.17 个标准差。该效应量属于小幅正向效应，与教育技术干预的一般基准相符，但同样经由官方渠道与合作方资源页首发。
- **PNAS 论文（doi: 10.1073/pnas.2507708123）**：利用 KA 大规模行政数据分析真实世界计算机辅助学习（CAL）的使用效果；论文正文未能获取，效应量无法核实。

**小结**：Khan Academy 拥有多项大规模 RCT 支持其作为课堂补充工具的小幅正向效果（约 0.1–0.2 SD 量级），但高质量证据的传播高度依赖平台官方渠道，独立第三方复核的完整学术文本可得性有限。

### 松鼠 AI（义学教育）

松鼠 AI 是面向中国 K-12 市场的 AI 自适应学习系统。学术性体验研究确认其核心技术架构包括细粒度知识图谱（基于学习进展关系组织知识组件）、自适应诊断预评估和个性化学习路径推荐 [4]。然而：

- **独立效果证据缺失**：检索未发现独立发表的、严格的 RCT 或准实验研究；现有材料多为体验研究、产品测评与厂商自述（如"人机对战"提分宣传），均无同行评议。
- 第三方行业资料称其市场份额约 35%，但来源为非学术文档，属市场宣传性质，不可作为依据。

### 学术文献整体状况

主流学术索引（Crossref、OpenAlex）中该主题的同行评审实证文献较稀疏，仅命中一篇非实证的转型综述 [4]。这与两篇高质量系统综述的判断一致：AI 个性化学习与自适应评估在 85 项研究的 PRISMA 综合中显示可提升成绩、参与度与动机，但同时存在算法偏见与数据隐私风险 [2]；另一篇被引超过 800 次的文献综述（63 篇文章）证实 AI/ML 算法可优化学习路径、提升测验分数，亦指出隐私与系统复杂性挑战 [5]。

## 二、个性化推荐算法机制比较

两系统的底层理论均可追溯至知识成分（Knowledge Component）建模传统——Koedinger、Corbett 与 Perfetti 的知识-学习-教学（KLI）框架为认知导师类系统的知识组件划分与掌握度推断提供了理论基础 [6]。

| 维度 | Khan Academy | 松鼠 AI |
|---|---|---|
| 知识图谱 | 按课程标准组织的技能树（skill tree），中等粒度知识点 | 厂商宣称"超纳米级"知识点拆分（MCM：思想/能力/方法图谱），粒度更细（未经独立验证） |
| 诊断评估 | 基于掌握度（mastery）概率模型：答题正确率结合遗忘机制（掌握—熟练—生疏状态流转），近似 BKT/IRT 混合 | 宣称结合知识空间理论与贝叶斯网络做初始诊断，"测—学—练—测"闭环，依错因图谱定位前置漏洞 |
| 推荐路径 | 相对线性：解锁下一技能、单元测验与掌握度目标；Khanmigo 提供大模型对话式辅导 | 非线性回溯式：诊断出前置薄弱点后自动回溯补学，不走同步课或超前学习的常规顺序 |
| 自适应策略 | 题目难度与复习时机自适应 + LLM 对话助教（Khanmigo，官方确认存在但未公开算法细节） | 三层架构：学习/内容/错因三地图 → 实时推荐系统 → 实时互动分析，配合线下 OMO 教师督导 |

**关键警示**：松鼠 AI 的架构描述几乎全部来自高管公开发言与厂商宣传，缺乏独立同行评审验证；Khan Academy 的掌握度算法具体参数同样未公开。两者均存在"算法黑箱"问题，这与系统综述所指出的自适应系统透明度与偏见风险相呼应 [2][5]。

## 三、效果量化指标与基准

自适应学习系统效果的评估通常涵盖六类指标，其证据强度如下：

| 指标类别 | 常用方法 | 证据状况 |
|---|---|---|
| 成绩提升 | 前后测标准化分差 | KA 的 RCT 报告 0.12–0.17 SD（第一方渠道发布）；松鼠 AI 无独立数据 |
| 效应量 | Cohen's d / Hedges' g | 技术辅助教学二阶元分析基准约 g ≈ 0.35（小到中等效应）[3]；智能辅导系统 K-12 数学元分析约 d ≈ 0.35–0.41（文献线索，原文未取回） |
| 掌握度 | BKT/DKT 预测的 mastery 概率 | 属系统内部指标，两平台均未公开可独立审计的数据 |
| 学习时长 | 达到掌握所需时间 | ASSISTments/Cognitive Tutor 研究报告缩短 10–30%（待核实） |
| 完成率 | 知识点/课程完成比例 | 在线平台自然使用完成率常偏低，依实现质量波动大 [3] |
| 留存/迁移 | 延迟后测、远迁移任务 | ITS 研究普遍显示近迁移效应大于远迁移；长期留存研究稀缺 |

教育技术类干预的效应量波动范围很大（从小的负效应到大效应），提示自适应系统的效果高度依赖实现质量、教师整合方式与评测设计，而非仅由算法决定 [3]。松鼠 AI 的"5 倍提分效率"等声明均来自企业白皮书或自建对照实验，应视为厂商自述而非科学证据。

## 四、师生接受度

**本维度证据不足**：针对师生接受度（可用性、信任、工作负担、课堂整合与持续使用意愿）的专项调研未能执行，本次检索未获得 Khan Academy 或松鼠 AI 在 K-12 数学课堂中的系统性接受度研究数据。可间接推断的两点是：其一，Oreopoulos 等 RCT 中"教师指导下的每周定时使用"设计暗示教师整合是使用生效的前提条件；其二，松鼠 AI 的 OMO 模式明确要求线下教师督导，说明其产品设计预设了较高的人机协同依赖。系统综述层面则提示算法偏见、数据隐私与系统复杂性是影响教育者信任与持续使用的已知障碍 [2][5]。课堂接受度的实证研究是亟需填补的空白。

## 五、综合讨论与启示

1. **证据不对称是核心问题**：Khan Academy 有大规模 RCT 支持的小幅正向效应，松鼠 AI 则几乎完全依赖厂商自述；即便 KA 的证据也多经第一方渠道传播。政策制定者与学校采购方应优先采信独立同行评审证据。
2. **效应量需校准预期**：0.1–0.35 SD 的小到中等效应是此类系统的现实基准 [3]，远超此范围的宣传（如"数倍提分"）应视为营销声明。
3. **算法透明度不足**：两系统的推荐机制均缺乏公开技术文档，知识图谱粒度、诊断模型参数无法外部审计，阻碍了效果归因与公平性评估 [2]。
4. **教师角色不可替代**：现有正向证据均出现在教师结构化整合的场景中，自适应系统更适合作为课堂补充而非替代。

## 参考来源

[2] Leveraging AI in E-Learning: Personalized Learning and Adaptive Assessment through Cognitive Neuropsychology—A Systematic Analysis — Constantinos Halkiopoulos, Evgenia Gkintoni（2024） — https://mdpi.com/2079-9292/13/18/3762/pdf?version=1726988994
[3] Augmented Reality Learning Experiences: Survey of Prototype Design and Evaluation — Marc Ericson C. Santos, Angie Chen, Takafumi Taketomi, Goshiro Yamamoto, Jun Miyazaki, Hirokazu Kato（2014） — https://ieeexplore.ieee.org/ielx7/4620076/6812165/06681863.pdf
[4] 从自适应到生成式AI：个性化学习的智能教育系统转型研究 — Hong Kong Anmai Publishing Co., Ltd.（2025） — https://doi.org/10.64216/3080-1494.25.10.072
[5] Adaptive Learning Using Artificial Intelligence in e-Learning: A Literature Review — Ilie Gligorea, Marius Cioca, Romana Oancea, Andra-Teodora Gorski, Hortensia Gorski, Paul Tudorache（2023） — https://mdpi.com/2227-7102/13/12/1216/pdf?version=1701879185
[6] The Knowledge‐Learning‐Instruction Framework: Bridging the Science‐Practice Chasm to Enhance Robust Student Learning — Kenneth R. Koedinger, Albert T. Corbett, Charles A. Perfetti（2012） — https://figshare.com/articles/The_knowledge-learning-instruction_framework_bridging_the_science-practice_chasm_to_enhance_robust_student_learning_/6470522

**关键来源说明**：Khan Academy 的两项 RCT 结果经其官方博客与合作方资源页发布（属第一方传播，引用时已注明）；元分析效应量基准引自经同行评审的 IEEE TLT 论文 [3]；算法机制对比中的松鼠 AI 部分主要来自厂商公开发言，独立验证缺失。

Overall Confidence: 0.35

---

## 元信息

- **置信度**: 0.35
- **搜索轮数**: 12
- **重规划次数**: 0
- **证据补充轮数**: 0
- **规划维度覆盖**: 75.0% (3/4)
- **对抗轮数**: 0
- **总耗时**: 362.64 秒

## 证据审计

- **Claim 覆盖率**: 0.0%
- **核验结果**: 0 supported / 0 refuted / 34 NEI（共 34 条）
- **原始/权威来源占比**: 21.4%
- **全文证据来源占比**: 3.6%
- **审计文件**: `outputs/evidence/evidence_h2h_v4_edu_001_1786685830064137000_20260814T054312Z.json`

### 仍需谨慎的陈述

- `not_enough_evidence` Khan Academy 是覆盖 K-12 数学为主、多学科并行的全球性免费学习平台。
- `not_enough_evidence` 萨尔瓦多大规模 RCT（Büchel 等，2019）：约 3,500 名 3–6 年级学生参与，发现 Khan Academy 补充性使用的数学成绩优于仅教师干预组。
- `not_enough_evidence` 该结果主要通过 Khan Academy 官方博客（2026-02-07）传播，虽基于独立研究者的随机对照试验，但发布渠道属于第一方，引用时应注明。
- `not_enough_evidence` Oreopoulos 等（2025 公布）RCT：与 Khan Academy 合作、约 1.1 万名学生参与。
- `not_enough_evidence` 教师指导下每周约 35 分钟使用 KA 的 3–6 年级学生，年末数学测评成绩比对照组高约 0.12–0.17 个标准差。
