🧠 GrowableLLM: 动态生长与终身学习大模型实验框架本项目是一个极具实验性质的自回归语言模型（LLM）框架，旨在探索大模型的“终身学习”（Lifelong Learning）与“脑区物理扩容”机制。
通过打破传统 LLM 训练后参数固定的局限，本项目允许模型在不遗忘旧有知识的前提下，通过动态增加前馈神经网络（FFN）的维度来学习新领域的知识（如：高情商对话、医学专家等）。  
✨ 核心特性动态扩容 (Dynamic Expansion): 运行时物理增加神经元，支持零初始化（Zero-Initialization），新突触平滑接入。  
完美梯度锁 (Perfect Gradient Lock): 彻底隔绝灾难性遗忘。扩容后通过底层的 Hook 机制，精准冻结基座旧脑区的梯度，确保每次特训只影响新长出的神经元。  
小乐协议 - 碎片整理 (Defrag Scheme B): 独特的经验回放与不对称解锁机制，实现新旧认知网络的高效融合。  
无缝继承开源智慧: 拥有一键“器官移植”脚本，直接提取官方 Qwen1.5-0.5B 的参数矩阵作为初始底座，免去昂贵的从零预训练成本。  
🛠️ 环境准备与安装本项目依赖的库已在 requirements.txt 中列出。建议使用 Python 3.8+ 及支持 CUDA 的环境进行实验。
克隆仓库
Bash
    git clone https://github.com/jack66g/Growable-LLM.git
    cd  Experiment_Replication
安装依赖
Bash
    pip install -r requirements.txt
🚀 实验复现指南
请严格按照以下步骤执行，以复现从“纯净基座提取”到“高情商脑区生长”的全过程。
阶段一：提取并重铸初始大脑矩阵我们的实验不从零开始，而是基于 HuggingFace 上的 Qwen/Qwen1.5-0.5B 提取参数，并完美映射到我们的 GrowableLLM 架构中。
运行指令：
Bash
     python convert_qwen_to_growable.py
发生了什么？ 脚本会自动下载 Qwen 的权重，将 24 层 Transformer 块的注意力层（Attention）和全连接层（MLP）无缝移植到本项目的动态框架中。
预期产物： 运行成功后，会在当前目录生成一个名为 growable_qwen_base.pth 的核心权重文件。
阶段二：定向生长“高情商对话”脑区
在拥有了具备通用世界观的基座后，我们将为其注入第一层专业技能。这一步会触发模型的动态扩容机制。
⚠️ 前置准备： 确保你的项目根目录下存在用于训练的高情商对话语料文件 daily_chat_clean.jsonl。
运行指令：Bash
     python train_chat_expert.py
发生了什么？
模型读取 growable_qwen_base.pth 并加载通用认知。
触发 expand_model(extra_dim=256)，为所有 24 层 FFN 动态长出 256 维的空白神经元。
启动完美梯度锁，0.5B 基座原本的常识被绝对冻结，梯度仅注入新增的 256 维突触。
使用 ChatML 格式和混合精度训练（AMP）跑满 3 个 Epoch。  预
期产物： 训练完成后，将固化并保存新权重为 growable_chat_expert_epoch3.pth。
📂 核心文件索引model.py：框架的核心引擎。包含 GrowableLLM 的网络定义、DynamicSwiGLU 动态扩容模块、RoPE 修正以及用于长文本生成的 KV Cache 机制。
convert_qwen_to_growable.py：权重无损转换与映射脚本。
train_chat_expert.py：第一阶段微调与扩容的实战脚本。内含针对 Windows 多进程 DataLoader 优化过的 ChatCollate 批处理类。
🔮 下一步（Coming Soon）
目前的框架已成功完成第一阶段的生长。在后续的更新中，我们将：
引入更复杂的医学脑区数据进行二次扩容。
激活并测试核心的 model.defrag() 方法，验证“不对称碎片整理”在多领域技能融合下的真实表现。  
