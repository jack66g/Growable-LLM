<h1 align="center">GrowableLLM</h1>

<p align="center">
<a href="README.md">English</a> | <a href="README_CN.md">简体中文</a>
</p>

<p align="center"><a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a> <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.5%2B-orange" alt="PyTorch"></a> <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/uv-managed-purple" alt="uv"></a></p>

<p align="center"><a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a> <a href="https://github.com/jack66g/Growable-LLM/issues"><img src="https://img.shields.io/github/issues/jack66g/Growable-LLM" alt="GitHub issues"></a> <a href="https://github.com/jack66g/Growable-LLM"><img src="https://img.shields.io/github/stars/jack66g/Growable-LLM" alt="GitHub stars"></a></p>

<p align="center"><em>通过正交突触扩展，让 LLM 在不回放旧数据的前提下动态扩展隐藏单元，持续学习新领域知识。</em></p>

---

## 项目概述

传统 LLM 训练完成后参数即被固定，学习新知识必然导致灾难性遗忘（Catastrophic Forgetting）。GrowableLLM 另辟蹊径：**不为模型"覆盖重写"，而是动态扩展模型容量。**

核心思路是运行时动态增加 FFN 维度（最多扩展至 7B 参数量级），通过精确梯度锁冻结旧参数，让新知识只写入新长出的隐藏单元，从而实现无回放的持续预训练。

---

## 核心特性

- **动态扩容（Dynamic Expansion）**：运行时动态增加 FFN 隐藏单元维度。新权重随机初始化（Kaiming uniform），前向输出因新增维度未参与计算而保持不变，梯度从第一次 backward 开始正常流动。
- **精确梯度锁（Exact Gradient Lock）**：通过 PyTorch register_hook 机制，精准冻结已有参数区域的梯度，确保每次训练只影响新长出的隐藏单元，有效遏制灾难性遗忘。
- **参数对齐（Parameter Alignment, Scheme B）**：不对称解锁机制——临时解锁顶层 6 层 Attention 模块与最终 RMSNorm，以极低学习率（1e-5）回放 replay buffer 中缓存的少量 token 数据进行微调，使新隐藏单元适应旧有注意力分布，实现新旧知识融合并最大限度保留旧表示。
- **即插即用基座**：支持从 HuggingFace 提取 Qwen1.5-0.5B / SmolLM2-360M 等模型权重，直接映射到 GrowableLLM 架构。

---

## 与 PNN 的关系

GrowableLLM 的设计思想深受 **Progressive Neural Networks (PNN)** 启发，但在实现路径上有本质差异：

| 维度 | PNN | GrowableLLM |
|------|-----|-------------|
| **扩容单位** | 每学一个新任务，新增一整个独立的网络"列"（column） | 每学一个新领域，在每层 FFN 内部增量增加 hidden 维度 |
| **参数增长** | 参数量随任务数线性倍增（每列 ~0.5B，2 个任务即翻倍） | 参数量随领域数微增（每领域 ~10M，粒度 128~256 dim） |
| **跨任务交互** | 旧列通过 lateral connection 向新列提供特征 | 新旧知识共享同一网络实体，通过梯度掩码实现逻辑隔离 |
| **推理成本** | 所有列需同时参与推理，延迟随任务数线性增长 | 推理时网络规模不变，可选择性屏蔽已有参数区域 |
| **计算效率** | FLOPs 随任务数线性增长 | FLOPs 增长仅来自新增维度（新增隐藏单元参与前向计算），无额外列级开销 |
| **核心机制** | lateral connection + 冻结旧列 | 正交突触扩展 + 精确梯度锁 + 参数对齐 |

简言之，**PNN 跨列扩展；GrowableLLM 层内扩展。** 前者适合任务边界清晰的场景，后者更接近增量式表示学习范式，在 LLM 的连续预训练场景下参数效率更高。

---

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch >= 2.5.0
- CUDA 12.1（推荐）
- [uv](https://docs.astral.sh/uv/)（包管理器）

### 安装

```bash
git clone https://github.com/jack66g/Growable-LLM.git
cd Growable-LLM
uv sync                                    # 核心依赖
uv sync --extra experiment --extra benchmark  # 全部依赖（含实验 + 基准测试）
```


---

## 项目结构

<details>
<summary>点击展开项目结构</summary>

```
Growable-LLM/
├── models.py                     # 核心模型定义（GrowableLLM, DynamicSwiGLU）
├── stats.py                      # CSV 统计分析工具
├── outputs/                      # 实验输出统一目录
│   ├── catastrophic_forgetting/
│   ├── continual_benchmark/
│   ├── cross_interference/
│   ├── defrag_benchmark/
│   ├── expansion_efficiency/
│   ├── hooklock_final/
│   ├── knowledge_localization/
│   ├── neural_age/
│   └── scaling_law/
├── Experiment/                   # 8 个消融实验与基准测试
│   ├── Ablation.py               # 神经年龄消融
│   ├── Forgetting.py             # 灾难性遗忘压力测试
│   ├── Localization.py           # 知识定位基准
│   ├── Matrix.py                 # 交叉干扰测试（10 领域）
│   ├── Performance.py            # 扩容效率基准
│   ├── Scaling_Law.py            # Scaling Law 拟合
│   ├── benchmark_fusion.py       # Defrag 必要性对比
│   ├── hook.py                   # Hook Lock A/B 测试
│   ├── memory_test.py            # 持续学习基准（BWT/FWT）
│   └── discarded/                # 已归档实验脚本
├── Experiment_Replication/       # Qwen1.5-0.5B 复现流水线
│   ├── convert_qwen_to_growable.py   # 权重提取与转换
│   ├── train_chat_expert.py          # 阶段一：聊天角色（Social QA）训练
│   ├── train_med_expert.py           # 阶段二：医疗角色（Medical QA）训练 + 参数对齐
│   ├── test_chat_expert.py           # 聊天角色测试终端
│   └── test_med_expert.py            # 双角色热切换测试终端
├── Finalized_Test/              # SmolLM2-360M 标准化评测流水线
│   ├── config.json                   # 统一配置（硬件配置、训练参数、路径）
│   ├── extract_weights.py            # 基座权重提取与转换
│   ├── download_datasets.py          # 训练数据下载
│   ├── train_pipeline.py             # 两阶段扩容训练 + 参数对齐
│   ├── test_baseline.py              # 基座模型交互式测试
│   ├── test_master.py                # 扩容后模型交互式测试
│   └── run_benchmark.py              # WikiText-2 PPL + GSM8K 评测
```
</details>

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch |
| 模型架构 | SwiGLU FFN + RoPE + GQA Attention |
| 动态扩容 | 自定义 DynamicSwiGLU.expand() |
| 梯度隔离 | PyTorch register_hook |
| 基座模型 | Qwen1.5-0.5B / SmolLM2-360M-Instruct |
| 依赖管理 | uv / pyproject.toml |

---

## 相关项目

- [Qwen1.5-0.5B] — 基座模型
- [SmolLM2-360M-Instruct] — 基座模型

---

## 联系

有 Bug 或建议欢迎联系：**18200793117@163.com** | **tbnl_zldyd@outlook.com**

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jack66g/Growable-LLM&type=Date)](https://star-history.com/#jack66g/Growable-LLM&Date)

---

## License

本项目采用 Apache License, Version 2.0。

[Qwen1.5-0.5B]: https://huggingface.co/Qwen/Qwen1.5-0.5B
[SmolLM2-360M-Instruct]: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct