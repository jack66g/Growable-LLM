# GrowableLLM: 动态生长与终身学习大模型实验框架

> 探索大模型"终身学习"（Lifelong Learning）与"脑区物理扩容"机制的实验框架。通过正交突触扩展，让 LLM 在不回放旧数据的前提下动态生长神经元，持续学习新领域知识。

---

## 项目概述

传统 LLM 训练完成后参数即被固定，学习新知识必然导致灾难性遗忘（Catastrophic Forgetting）。GrowableLLM 另辟蹊径：**不为模型"覆盖重写"，而是为它物理生长新脑区。**

核心思路是运行时动态增加 FFN 维度（最多扩展至 7B 参数量级），通过完美梯度锁冻结旧参数，让新知识只写入新长出的神经元，从而实现无回放的持续预训练。

---

## 核心特性

- **动态扩容（Dynamic Expansion）**：运行时物理增加 FFN 神经元维度。新突触零初始化接入，前向输出不变，梯度从第一次 backward 开始正常流动。
- **完美梯度锁（Perfect Gradient Lock）**：通过 PyTorch register_hook 机制，精准冻结基座旧脑区的梯度，确保每次特训只影响新长出的神经元，彻底隔绝灾难性遗忘。
- **碎片整理（Defrag Scheme B）**：不对称解锁机制——临时解锁顶层 Attention 模块，以极低学习率回放少量隐藏状态，实现新旧知识融合而不覆盖旧表示。
- **即插即用基座**：支持从 HuggingFace 提取 Qwen1.5-0.5B / SmolLM2-360M 等模型权重，无缝映射到 GrowableLLM 架构。

---

## 与 PNN 的关系

GrowableLLM 的设计思想深受 **Progressive Neural Networks (PNN)** 启发，但在实现路径上有本质差异：

| 维度 | PNN | GrowableLLM |
|------|-----|-------------|
| **扩容单位** | 每学一个新任务，新增一整个独立的网络"列"（column） | 每学一个新领域，在每层 FFN 内部增量增加 hidden 维度 |
| **参数增长** | 参数量随任务数线性倍增（每列 ~0.5B，2 个任务即翻倍） | 参数量随领域数微增（每领域 ~10M，粒度 128~256 dim） |
| **跨任务交互** | 旧列通过 lateral connection 向新列提供特征 | 新旧知识共享同一网络实体，通过神经元年龄/梯度锁物理隔离 |
| **推理成本** | 所有列需同时参与推理，延迟随任务数线性增长 | 推理时网络规模不变，物理路由可选择性"断电"旧脑区 |
| **核心机制** | lateral connection + 冻结旧列 | 正交突触扩展 + 完美梯度锁 + Defrag 碎片整理 |

简言之，**PNN 是横向"叠层"——加整列；GrowableLLM 是纵向"生长"——扩单层。** 前者适合任务边界清晰的场景，后者更贴近生物大脑的渐进式学习，在 LLM 的连续预训练场景下参数效率更高。

---

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.7+（推荐）

### 安装

```bash
git clone https://github.com/jack66g/Growable-LLM.git
cd Growable-LLM
pip install -r Experiment_Replication/requirements.txt
```

### 最小运行示例（模型自测试）

```bash
python models.py
```

该命令会执行模型定义中的 `__main__` 自测试：构建 GrowableLLM → 前向传播 → 动态扩容 → 梯度锁验证 → 碎片整理 → 文本生成 → 检查点保存/加载。

---

## 项目结构

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
│   ├── train_chat_expert.py          # 阶段一：聊天脑区训练
│   ├── train_med_expert.py           # 阶段二：医学脑区训练 + Defrag
│   ├── test_chat_expert.py           # 聊天人格测试终端
│   └── test_med_expert.py            # 双人格热切换测试终端
├── Finalized_Test/
│   ├── extract_weights.py
│   ├── download_datasets.py
│   ├── train_pipeline.py
│   ├── test_baseline.py
│   ├── test_master.py
│   └── run_benchmark.py
```

---

## 使用指南

### 实验复现（Qwen1.5-0.5B）

三步复现从基座提取到双领域专家训练的完整流程：

```bash
cd Experiment_Replication

# 步骤 1：提取 Qwen1.5-0.5B 权重
python convert_qwen_to_growable.py
# → 输出：growable_qwen_base.pth

# 步骤 2：训练高情商对话脑区（扩容 256 维）
python train_chat_expert.py
# → 输出：growable_chat_expert_epoch3.pth

# 步骤 3：训练老中医脑区（扩容 128 维 + Defrag 融合）
python train_med_expert.py
# → 输出：growable_med_expert_epoch3.pth

# 步骤 4：双人格热切换测试
python test_med_expert.py
# → 控制台输入 /chat 或 /med 切换人格
```
---

## 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch |
| 模型架构 | SwiGLU FFN + RoPE + GQA Attention |
| 动态扩容 | 自定义 DynamicSwiGLU.expand() |
| 梯度隔离 | PyTorch register_hook |
| 基座模型 | Qwen1.5-0.5B / SmolLM2-360M-Instruct |
| 依赖管理 | pip / requirements.txt |

---

## 相关项目

- [Qwen1.5](https://huggingface.co/Qwen/Qwen1.5-0.5B) — 基座模型

---


## 联系

有 Bug 或建议欢迎联系：**18200793117@163.com**
