🧬 Neural Age Ablation Benchmark (神经元生命周期实验)

这是一个针对模型持续学习能力的消融实验。主要用于验证动态扩容后的网络中，神经元是否会随着“年龄”增长自发降低可塑性，并最终固化为稳定的长期记忆层，以此来探索认知模型中 $e^{-\lambda \text{age}}$ 的理论衰减规律。

实验运行结束后，系统会在 neural_age_outputs/ 目录下自动输出完整的数据统计表 (CSV) 以及可塑性衰减、梯度变化、特征覆盖等各项指标的可视化图表。

🚀 快速启动

在终端中直接运行以下命令即可触发完整评测：

Bash

python Ablation.py

🧩 Defrag Necessity Benchmark (碎片整理机制实验)

这是一个验证“不对称碎片整理 (Defragmentation)”必要性的对比实验。通过在 10 个不同的知识领域进行连续学习，代码将横向对比三种策略的表现：

V1: 完全不进行碎片整理 (No Defrag)

V2: 顶层完全解锁融合 (Full Defrag)

V3: 仅解锁顶层注意力机制 (Attention Defrag)

实验的核心目的是找出哪种策略能在保证最低遗忘率 (Forgetting %) 和交叉干扰的前提下，最大化对未来未知任务的前向泛化能力 (FWT Gain)。运行结束后，系统会在 defrag_benchmark_results/ 目录下自动生成 10x10 交叉干扰热力图、遗忘率曲线等详细的可视化图表。

🚀 快速启动

在终端中直接输入以下命令即可触发完整对比测试：

Bash

python benchmark_fusion.py

💣 Catastrophic Forgetting Stress Test (灾难性遗忘压力测试)

这是一个极端的“强冲突”连续学习基准测试。通过人工构造多组输入极度相似、但输出目标完全冲突的任务（例如 Task A 中 token 10 指向 20，Task B 中 token 10 指向 999），来极限施压模型的底座认知。

实验主要用于验证 GrowableLLM 架构在极端的知识覆写场景下，能否通过“完美梯度锁”和“碎片整理机制”物理隔离新旧知识，避免特征表征崩溃，实现双重记忆共存（Dual-Memory Coexistence）。运行结束后，系统会在 catastrophic_forgetting_outputs/ 目录下生成遗忘矩阵热力图、知识留存曲线等关键指标图表。

🚀 快速启动

在终端中直接运行以下命令即可触发压力测试：

Bash

python Forgetting.py

🔒 Hook Lock Stability Benchmark (完美梯度锁消融实验)

这是一个针对“完美梯度锁 (Perfect Gradient Lock)”机制的核心 A/B 测试。实验将在 8 个连续的知识领域中，横向严格对比不加锁（NoLock）与加锁（HookLock）状态下，模型在动态扩容时的表现差异。

它的核心目的是通过量化“旧记忆漂移 (Old Memory Drift)”和“激活稀疏度 (Routing Sparsity)”，从物理层面证明通过 PyTorch Hook 掩码能否绝对冻结已固化的旧突触权重，从而实现真正意义上的新旧知识隔离。运行结束
后，系统会在 hooklock_final/ 目录下生成直观的双线对比曲线（如 drift.png 漂移对比, forgetting.png 遗忘对比）以及效果反差强烈的交叉验证热力图。

🚀 快速启动

在终端中直接运行以下命令，即可一键触发完整的对比评测：

Bash

python hook.py

🚨 True No-Lock Baseline (灾难性遗忘对照基准)

这是一个极其关键的绝对对照组 (Baseline) 实验。它在连续学习 8 个不同领域数据时，完全关闭了梯度锁 (Hook Lock)、特征路由 (Routing) 以及数据回放 (Replay) 保护，仅保留了基础的动态扩容与全参数微调。  

本实验的核心目的在于“反向验证”：通过观测模型在这种无保护状态下必然发生的旧记忆参数漂移 (Old Memory Drift) 与灾难性遗忘 (Catastrophic Forgetting)，来直接证明你设计的 Hook Lock 机制的有效性与不可或缺性。运行结束后，灾难性遗忘热力图、任务性能衰减矩阵等文件会自动保存在 true_nolock_baseline/ 目录下
。 
🚀 快速启动

该文件命名为 no_hook.py，只需在终端中直接运行以下命令即可触发对照组评测：

Bash

python no_hook.py

📍 Knowledge Localization Benchmark (知识局部化验证)

这是一个验证模型内部“脑区功能分化”现象的观测实验。在赋予大语言模型持续学习能力后，我们需要确切地知道：不同的垂直领域知识（如医学、代码、数学等）是否会自动局部化（Localize）到特定的神经元簇中，而不是像传统网络那样发生特征的全局弥散。

本实验通过 PyTorch Hooks 提取模型在 8 个不同领域下的 FFN 激活签名，重点计算神经元重合度 (Overlap Score) 和路由熵 (Routing Entropy)，以此来评估认知分化的程度。运行结束后，系统会在 knowledge_localization_outputs/ 目录下自动生成直观的 t-SNE 聚类图、神经元特化曲线以及领域重叠热力图。

🚀 快速启动

在终端中直接运行以下命令即可提取特征并生成可视化图表：

Bash

python Localization.py

🕸️ Cross-Interference Benchmark (跨领域交叉干扰实验)

这是一个极其敏锐的基准测试，专门用来量化模型在连续吸收多领域知识时产生的“知识污染” (Cross-Interference) 现象。

这份代码最出彩的地方在于，它巧妙地应用了合成数据生成 (Synthetic Data Generation) 技术来构造测试集：通过设定全局共享的底层 Token 规律（base_pattern）和特定领域的偏移量（domain_shift），强制在模型的隐空间（Latent Space）中制造特征重叠。

它的核心评测逻辑是计算干扰值：$I(i,j) = \text{Loss}_{\text{after\_j}}(i) - \text{Loss}_{\text{after\_i}}(i)$。如果该值大于 0，就确凿地证明了刚刚学习的新领域 $j$ 破坏了模型对旧领域 $i$ 的记忆。测试完成后，系统会在 cross_interference_outputs/ 目录下输出极具视觉冲击力的 10x10 交叉干扰热力图和各领域的知识留存曲线。

🚀 快速启动在终端中直接运行以下命令，即可一键触发包含 10 个领域的交叉干扰矩阵评测：

Bash

python Matrix.py

🏆 Continual Domain Benchmark (综合持续学习评估套件)

这是整个项目中最具“顶会 Paper 质感”的终极基准测试。它将前面验证过的所有机制（动态扩容、完美梯度锁、无回放碎片整理）整合在一起，让模型在 10 个不同的知识领域中进行真正的连续学习实战。

除了直观的 Loss 追踪，这份代码最大的亮点在于它原生集成了大模型前沿研究中的核心学术指标，非常适合用来展示大模型 (LLM) 底座架构的实际工程价值：

BWT (Backward Transfer, 向后转移)：精确量化学习新知识后，对历史旧知识的记忆保留度（越接近 0 越好，负数代表遗忘）。

FWT (Forward Transfer, 向前转移)：评估模型底座在扩容后，对未来尚未学习的全新领域的泛化与加速能力。

AVG ACC (Average Accuracy)：训练周期结束后的全局平均性能。

测试完成后，系统会自动在 continual_benchmark_outputs/ 目录下生成高质量的综合性能统计表 (CSV)、各领域的记忆留存曲线以及跨领域干扰热力图，图表可直接用于技术汇报或开源展示。

🚀 快速启动

在终端中直接运行以下命令，即可启动这一套完整的端到端综合评测：

Bash

python memory_test.py

📈 Expansion Efficiency Benchmark (动态扩容效能与 Scaling Law 验证)

这是一个用于评估模型“参数投资回报率”的核心基准测试。在 GrowableLLM 进行动态扩容时，我们需要确切知道：无脑增加参数是否永远有效？模型何时会触及边际收益递减（Diminishing Returns）？

本实验通过设定一系列递增的扩容阶梯（如 +64, +128, ..., +1024 维度），在多领域合成数据集上持续训练，并严密监控核心指标：扩容效能 (Efficiency)，即 $\frac{\Delta \text{Performance}}{\Delta \text{Params}}$。

运行结束后，系统会在 expansion_efficiency_outputs/ 目录下自动生成极具学术价值的可视化图表，包括模型专属的 Scaling Law 曲线（参数量 vs Loss）、扩容边际收益柱状图以及效能衰减曲线。这能够直观地帮助你找到模型生长的“甜点区 (Sweet Spot)”。

🚀 快速启动在终端中直接运行以下命令，即可一键触发扩容效能评测：

Bash

python Performance.py

📏 Scaling Law Benchmark (神经扩展定律拟合实验)

这是一个极具理论深度的实验，直接对标大模型研发领域的核心准则——Scaling Law (扩展定律)。在利用合成数据 (Synthetic Data) 不断喂养并动态扩容模型的过程中，我们需要用严格的数学模型来量化它的终极进化潜力。

代码通过设定一系列从极限微调到大规模增长的扩容阶梯（+32 到 +1536 维度），利用 scipy.optimize 强行拟合幂律曲线 $L(N) = aN^{-\alpha} + b$。它能帮你精准计算出决定模型天花板的缩放指数 (Scaling Exponent, $\alpha$) 以及理论最低损失 (Irreducible Loss, $b$)。

更硬核的是，实验还会通过计算 Loss 下降的二阶导数，来捕捉模型容量发生突变的“相变点 (Phase Transition)”。运行结束后，系统会在 scaling_law_outputs/ 目录下生成高学术规格的 Log-Log 对数拟合图、幂律曲线和效能衰减图。这是证明动态生长的 LLM 架构具备长期潜力的完美理论背书。

🚀 快速启动在终端中直接运行以下命令，即可开始推演模型的 Scaling Law：

Bash

python Scaling_Law.py
