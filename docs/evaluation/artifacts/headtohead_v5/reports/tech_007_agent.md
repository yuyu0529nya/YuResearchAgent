# 研究报告：探讨量子计算在密码学（后量子密码 PQC）和药物发现领域的应用前景，分析 NIST 标准化进程和当前技术瓶颈。

---

# 量子计算的双重前景：后量子密码迁移与药物发现——NIST 标准化进程与技术瓶颈分析

## 一、两大应用领域对比概览

| 维度 | 后量子密码（PQC） | 药物发现（量子分子模拟） |
|---|---|---|
| 核心机制 | 防御性：Shor 算法威胁 RSA/ECC，需迁移至抗量子算法 [1][5] | 进攻性：VQE/QPE 直接模拟分子电子结构 [16] |
| 成熟度 | 高——2024 年首批 FIPS 标准发布，已进入强制迁移阶段 [10] | 低——仍处概念验证与混合管线阶段 [3] |
| 时间锚点 | NIST 2035 联邦迁移死线；CNSA 2.0 要求 2027 年 NSS 系统就绪 [14] | 商业化落地预计 2028–2032 年（数百逻辑比特） |
| 主要瓶颈 | 实现开销、密码敏捷性、长期密码分析不确定性 | 退相干、纠错开销、缺乏容错硬件 [4][12] |
| 证据等级 | 官方标准一手文件充分 | 学术可行性强，但"量子优势"未获独立验证 [16] |

## 二、PQC：威胁驱动与 NIST 标准化进程

**威胁根源。** Shor 算法使足够规模的量子计算机能够破解 RSA、Diffie-Hellman 与椭圆曲线密码（ECC），对网络基础设施构成根本性威胁 [1][5]。NIST 早在 2016 年的报告中即指出，需要遴选新型公钥算法以补充现行联邦标准 [9]。"先存后解"（harvest now, decrypt later）攻击模式意味着：数据的保密寿命加上迁移所需年限若超过密码相关量子计算机（CRQC）的出现时间，则今天加密的数据届时将被破解——这是迁移紧迫性的定量基础。

**标准化时间线。** NIST 经过三轮公开竞赛式评审（2016–2022），第三轮状态报告记录了 7 个决赛算法与 8 个备选算法的遴选过程 [13]。2024 年 8 月 13 日，NIST 正式发布首批三项标准 [10]：

- **FIPS 203（ML-KEM）**：基于 CRYSTALS-Kyber 的模格密钥封装机制，安全性建立在 Module-LWE 困难问题上，规定 ML-KEM-512/768/1024 三个参数集 [10][20]；
- **FIPS 204（ML-DSA）**：基于 CRYSTALS-Dilithium 的格签名方案；
- **FIPS 205（SLH-DSA）**：基于 SPHINCS+ 的无状态哈希签名，作为格算法之外的保守备份 [20]。

这一"格为主、哈希为备"的结构反映了对密码分析不确定性的对冲：2022 年 SIDH 攻击攻破第四轮候选 SIKE 的事件表明，即使是进入标准化后期的算法也可能被颠覆。此后 NIST 在第四轮 KEM 附加轮中选择 HQC 作为补充方案，并启动附加签名征集（on-ramp）以丰富算法多样性；Classic McEliece 等基于编码的保守方案虽未入选标准，仍是重要的研讨与储备对象 [2]。

**政策驱动力。** 2026 年 6 月白宫管理与预算办公室（OMB）发布备忘录 M-26-15《后量子密码迁移执行》，要求各联邦机构制定并向 OMB 及国家网络总监办公室提交 PQC 迁移计划 [14]。结合 CNSA 2.0 对国家安全系统 2027 年的要求与 2035 年联邦全面迁移的目标，以及 Google（2029 年内部目标）、Cloudflare、Apple iMessage PQ3、Signal PQXDH 等的混合模式部署，迁移已由标准发布阶段进入工程执行阶段。ISO/IEC 亦将 ML-KEM、XMSS/LMS 等纳入相应标准，全球技术路线趋同。

**迁移侧瓶颈。** 主要难点包括：PQC 算法密钥与签名尺寸显著大于 ECC，对带宽受限协议（TLS 握手、固件签名）造成实现开销；系统需具备"密码敏捷性"（crypto-agility）以应对未来算法替换；格基方案的长期密码分析强度仍需时间检验 [5]。

## 三、药物发现：真实的物理依据与未兑现的优势

**科学逻辑。** 药物发现的核心计算瓶颈是精确求解分子电子结构；经典方法（DFT、Hartree-Fock）在强关联电子体系和共价键断裂场景下精度系统性不足，而量子计算"以量子系统模拟量子系统"在原理上可提供指数加速 [16]。两条技术路线分工明确：VQE（变分量子本征求解器）适配当前 NISQ 含噪声设备，QPE（量子相位估计）精度更高但依赖容错量子计算机 [16][17]。

**代表性进展。** Li 等（2024）在 *Scientific Reports* 发表的混合量子计算管线是标志性工作：针对前药活化的 Gibbs 自由能剖面与共价键相互作用等真实药物设计任务，达到了与经典基准一致的精度 [3]。Kumar 等（2024）的综述系统梳理了量子分子模拟在癌症药物筛选与计算机辅助药物设计中的潜力，同时指出退相干、算法噪声与可重复性等瓶颈 [7]；Atalor 等（2023）探讨了量子分子模拟加速癌症药物筛选的路径，强调传统高通量筛选难以捕捉生物分子相互作用的量子本性 [15]。企业层面，Boehringer Ingelheim 与 Google Quantum AI、IBM 与 Cleveland Clinic/Moderna 的合作均为厂商自述进展，尚缺独立验证。

**审慎判断。** 必须强调：**迄今没有公认的商业级"量子优势"实例**——所有报道的用例或是概念验证，或与经典方法等效 [3][16]。Outeiral 等明确警告需警惕炒作 [16]。临床试验设计优化的量子算法仍停留在论文阶段；同时经典张量网络与 AI 方法的进步不断抬高量子优势的门槛。价值最可能首先体现在共价药物、金属酶活性中心等经典方法系统性失效的特定化学空间。

## 四、共同的技术瓶颈：纠错开销是分水岭

两大应用的落地时间由同一个变量决定——容错量子硬件的成熟度 [4][12]：

- **纠错开销**：当物理比特错误率约 0.1% 时，单个逻辑量子比特需数千个物理比特冗余编码；物理比特典型错误率为 10⁻²–10⁻⁴ [4]。
- **工程挑战**：宇宙射线引发的多比特突发错误可破坏表面码纠错，Q3DE 等架构改进属增量缓解而非根治 [8]；各硬件平台（超导、离子阱、光量子、硅色心）在退相干与可扩展性上各有障碍 [12][18][19]。
- **时间表分歧**：破解 RSA-2048 需数千逻辑比特（数百万物理比特），主流估计 CRQC 不早于 2030 年代出现；IBM 等厂商路线图指向 2029 年前后的容错节点，但属第一方声明，学界估计更保守。这一估算本身依赖较早的资源分析，2025–2026 年的纠错突破是否实质压缩了窗口，缺乏同行评议一手数据的确认。

由此得出分层判断：**PQC 迁移紧迫性高且上升**（纠错每前进一步，"先存后解"窗口就缩小一分，NIST 2035 与 Google 2029 构成双重锚点 [10][14]）；**药物发现的实质性量子落地大概率在 2028–2032 年**随数百逻辑比特级机器出现，并面临经典 AI 方法的持续竞争 [17]。

## 五、结论

量子计算在密码学与药物发现两个领域呈现镜像关系：前者是**防御性刚需**——标准已定（FIPS 203/204/205）、政策已出（OMB M-26-15、CNSA 2.0），瓶颈在工程迁移而非科学原理 [10][14]；后者是**进攻性期权**——物理依据扎实、管线已验证可行，但决定性优势依赖容错硬件，且需与快速进步的经典方法赛跑 [3][16]。两者共同的节拍器是纠错开销：它决定了 CRQC 的到来时点（从而决定 PQC 的死线），也决定了量子药物模拟从"等效验证"走向"不可替代优势"的时点。

## 参考来源

[1] Recent Advances in Post-Quantum Cryptography for Networks: A Survey — Engin Zeydan, Yekta Türk, Berkin Aksoy, S. Bugrahan Ozturk（2022） — https://zenodo.org/record/7276560
[2] NIST PQC Seminar #18: Classic McEliece: conservative code-based cryptography — NIST — https://nist.gov/video/nist-pqc-seminar-18-classic-mceliece-conservative-code-based-cryptography
[3] A hybrid quantum computing pipeline for real world drug discovery — Weitang Li, Zhi Yin, Xiaoran Li 等（2024） — https://nature.com/articles/s41598-024-67897-8.pdf
[4] An elementary review on basic principles and developments of qubits for quantum computing — Eunmi Chae, Joonhee Choi, Junki Kim（2024） — https://nanoconvergencejournal.springeropen.com/counter/pdf/10.1186/s40580-024-00418-5
[5] Exploring Post-Quantum Cryptography: Review and Directions for the Transition Process — Kanza Cherkaoui Dekkaki, Igor Tasic, Maria-Dolores Cano（2024） — https://mdpi.com/2227-7080/12/12/241/pdf?version=1732363457
[7] Recent Advances in Quantum Computing for Drug Discovery and Development — Gautam Kumar, Sahil Yadav, Aniruddha Mukherjee, Vikas Hassija, Mohsen Guizani（2024） — https://ieeexplore.ieee.org/ielx7/6287639/6514899/10466774.pdf
[8] Q3DE: A fault-tolerant quantum computer architecture for multi-bit burst errors by cosmic rays — Yasunari Suzuki 等（2022） — http://hdl.handle.net/2324/7332376
[9] Report on Post-Quantum Cryptography — Lily Chen, Stephen P. Jordan, Yi-Kai Liu, Dustin Moody, René Peralta, Ray Perlner, Daniel Smith-Tone（2016） — https://doi.org/10.6028/nist.ir.8105
[10] FIPS 203, Module-Lattice-Based Key-Encapsulation Mechanism Standard — NIST CSRC（2024） — https://csrc.nist.gov/pubs/fips/203/final
[12] Quantum Computing: Navigating the Future of Computation, Challenges, and Technological Breakthroughs — Qurban A. Memon, Mahmoud Al Ahmad, Michael Pecht（2024） — https://mdpi.com/2624-960X/6/4/39/pdf?version=1731750103
[13] Status report on the third round of the NIST Post-Quantum Cryptography Standardization process — Gorjan Alagic 等（2022） — https://doi.org/10.6028/nist.ir.8413
[14] Execution of the Migration to Post-Quantum Cryptography (M-26-15) — The White House / OMB（2026） — https://whitehouse.gov/wp-content/uploads/2026/06/M-26-15-Execution-of-the-Migration-to-Post-Quantum-Cryptography.pdf
[15] Harnessing Quantum Molecular Simulation for Accelerated Cancer Drug Screening — Salvation Ifechukwude Atalor, Onuh Mathew Ijiga, Joy Onma Enyejo（2023） — https://ijsrmt.com/index.php/ijsrmt/article/download/502/134
[16] The prospects of quantum computing in computational molecular biology — Carlos Outeiral, Martin Strahm, Jiye Shi, Garrett M. Morris, Simon C. Benjamin, Charlotte M. Deane（2020） — https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/wcms.1481
[17] Quantum Computer Systems for Scientific Discovery — Yuri Alexeev 等（2021） — https://knowledge.uchicago.edu/communities/physical-sciences-division
[18] Room-temperature photonic logical qubits via second-order nonlinearities — Stefan Krastanov 等（2021） — https://nature.com/articles/s41467-020-20417-4.pdf
[19] Scalable Fault-Tolerant Quantum Technologies with Silicon Color Centers — Stephanie Simmons（2024） — http://link.aps.org/pdf/10.1103/PRXQuantum.5.010102
[20] gmsm/docs/pqc.md — emmansun/gmsm (GitHub) — https://github.com/emmansun/gmsm/blob/develop/docs/pqc.md

Overall Confidence: 0.55

---

## 元信息

- **置信度**: 0.55
- **搜索轮数**: 16
- **重规划次数**: 0
- **证据补充轮数**: 0
- **对抗轮数**: 0
- **总耗时**: 233.05 秒

## 证据审计

- **Claim 覆盖率**: 2.5%
- **核验结果**: 1 supported / 0 refuted / 39 NEI（共 40 条）
- **原始/权威来源占比**: 37.9%
- **全文证据来源占比**: 3.5%
- **审计文件**: `outputs/evidence/evidence_h2h_v4_tech_007_1786654467523885000_20260813T205820Z.json`

### 仍需谨慎的陈述

- `not_enough_evidence` Shor 算法使足够规模的量子计算机能够破解 RSA、Diffie-Hellman 与椭圆曲线密码（ECC），对网络基础设施构成根本性威胁 。
- `not_enough_evidence` NIST 早在 2016 年的报告中即指出，需要遴选新型公钥算法以补充现行联邦标准 。
- `not_enough_evidence` "先存后解"（harvest now, decrypt later）攻击模式意味着：数据的保密寿命加上迁移所需年限若超过密码相关量子计算机（CRQC）的出现时间，则今天加密的数据届时将被破解——这是迁移紧迫性的定量基础。
- `not_enough_evidence` NIST 经过三轮公开竞赛式评审（2016–2022），第三轮状态报告记录了 7 个决赛算法与 8 个备选算法的遴选过程 。
- `not_enough_evidence` FIPS 203（ML-KEM）：基于 CRYSTALS-Kyber 的模格密钥封装机制，安全性建立在 Module-LWE 困难问题上，规定 ML-KEM-512/768/1024 三个参数集 ；
