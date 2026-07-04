<h1 align="center">GrowableLLM</h1>

<p align="center">
<a href="README.md">English</a> | <a href="README_CN.md">简体中文</a>
</p>

<p align="center"><a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a> <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.5%2B-orange" alt="PyTorch"></a> <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/uv-managed-purple" alt="uv"></a></p>

<p align="center"><a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a> <a href="https://github.com/jack66g/Growable-LLM/issues"><img src="https://img.shields.io/github/issues/jack66g/Growable-LLM" alt="GitHub issues"></a> <a href="https://github.com/jack66g/Growable-LLM"><img src="https://img.shields.io/github/stars/jack66g/Growable-LLM" alt="GitHub stars"></a></p>

<p align="center"><em>Dynamic orthogonal synaptic expansion for replay-free continual pre-training of LLMs.</em></p>

---

## Overview

Traditional LLMs suffer from catastrophic forgetting when learning new domains. GrowableLLM takes a different path: **instead of overwriting, we dynamically expand model capacity.**

The core idea is to dynamically expand the FFN dimensions at runtime (up to ~7B parameters), using exact gradient locking to freeze old parameters so that new knowledge is written exclusively into newly grown hidden units, enabling replay-free continual pre-training.

---

## Core Features

- **Dynamic Expansion**: Dynamically increase FFN hidden unit dimensions at runtime. New weights are randomly initialized (Kaiming uniform). Forward output remains unchanged since new dimensions are not yet active. Gradients flow normally from the first backward pass.
- **Exact Gradient Lock**: PyTorch `register_hook` mechanism precisely freezes gradients for existing parameter subspaces. Each training session affects only newly grown hidden units, effectively preventing catastrophic forgetting.
- **Parameter Alignment (Scheme B)**: Asymmetric unlocking — temporarily unlocks the top 6 attention layers and final RMSNorm, then fine-tunes with a very low learning rate (1e-5) on cached token data from the replay buffer, allowing new hidden units to align with existing attention distributions.
- **Plug-and-Play Base Models**: Supports weight extraction from HuggingFace (Qwen1.5-0.5B / SmolLM2-360M) and direct mapping to the GrowableLLM architecture.

---

## Relationship with PNN

GrowableLLM is inspired by **Progressive Neural Networks (PNN)**, but differs fundamentally:

| Dimension                        | PNN                                                  | GrowableLLM                                                                                                  |
| -------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Expansion Unit**         | Adds a full independent network "column" per task    | Incrementally increases hidden dimension within each FFN layer per domain                                    |
| **Parameter Growth**       | Linearly doubles with task count (~0.5B per column)  | Grows incrementally per domain (~10M per domain, 128–256 dim granularity)                                   |
| **Cross-Task Interaction** | Old columns provide features via lateral connections | Old and new knowledge share the same network entity, isolated via gradient masking                           |
| **Inference Cost**         | All columns participate, latency grows linearly      | Network size unchanged at inference; old parameter subspaces can be selectively disabled                     |
| **Compute Efficiency**     | FLOPs grow linearly with task count                  | FLOPs grow only from new dimensions (new hidden units participate in forward pass), no extra column overhead |
| **Core Mechanism**         | Lateral connections + frozen old columns             | Orthogonal synaptic expansion + exact gradient lock + Parameter Alignment                                    |

In short: **PNN expands across columns; GrowableLLM expands within layers.** The latter is closer to incremental representation learning and more parameter-efficient for continuous pre-training.

---

## Quick Start

### Requirements

- Python 3.10+
- PyTorch >= 2.5.0
- CUDA 12.1 (recommended)
- [uv](https://docs.astral.sh/uv/) (package manager)

### Installation

```bash
git clone https://github.com/jack66g/Growable-LLM.git
cd Growable-LLM
uv sync                                    # Core dependencies
uv sync --extra experiment --extra benchmark  # Full dependencies (experiments + benchmarks)
```

---

## Project Structure

<details>
<summary>Click to expand project structure</summary>

```
Growable-LLM/
├── models.py                     # Core model definition (GrowableLLM, DynamicSwiGLU)
├── stats.py                      # CSV statistics utility
├── outputs/                      # Unified experiment output directory
│   ├── catastrophic_forgetting/
│   ├── continual_benchmark/
│   ├── cross_interference/
│   ├── defrag_benchmark/
│   ├── expansion_efficiency/
│   ├── hooklock_final/
│   ├── knowledge_localization/
│   ├── neural_age/
│   └── scaling_law/
├── Experiment/                   # 8 ablation experiments and benchmarks
│   ├── Ablation.py               # Neural age ablation
│   ├── Forgetting.py             # Catastrophic forgetting stress test
│   ├── Localization.py           # Knowledge localization
│   ├── Matrix.py                 # Cross-interference test (10 domains)
│   ├── Performance.py            # Expansion efficiency benchmark
│   ├── Scaling_Law.py            # Scaling law fitting
│   ├── benchmark_fusion.py       # Defrag necessity comparison
│   ├── hook.py                   # Hook Lock A/B test
│   ├── memory_test.py            # Continual learning benchmark (BWT/FWT)
│   └── discarded/                # Archived experiment scripts
├── Experiment_Replication/       # Qwen1.5-0.5B replication pipeline
│   ├── convert_qwen_to_growable.py   # Weight extraction and conversion
│   ├── train_chat_expert.py          # Phase 1: Chat role (Social QA) training
│   ├── train_med_expert.py           # Phase 2: Medical role (Medical QA) training + Parameter Alignment
│   ├── test_chat_expert.py           # Chat role test terminal
│   └── test_med_expert.py            # Dual-role hot-switch test terminal
├── Finalized_Test/              # SmolLM2-360M standardized evaluation pipeline
│   ├── config.json                   # Unified config (hardware configuration, training params, paths)
│   ├── extract_weights.py            # Base weight extraction and conversion
│   ├── download_datasets.py          # Training data download
│   ├── train_pipeline.py             # Two-phase expansion training + Parameter Alignment
│   ├── test_baseline.py              # Baseline model interactive test
│   ├── test_master.py                # Expanded model interactive test
│   └── run_benchmark.py              # WikiText-2 PPL + GSM8K evaluation
```

</details>

---

## Tech Stack

| Component               | Technology                           |
| ----------------------- | ------------------------------------ |
| Deep Learning Framework | PyTorch                              |
| Model Architecture      | SwiGLU FFN + RoPE + GQA Attention    |
| Dynamic Expansion       | Custom DynamicSwiGLU.expand()        |
| Gradient Isolation      | PyTorch register_hook                |
| Base Models             | Qwen1.5-0.5B / SmolLM2-360M-Instruct |
| Dependency Management   | uv / pyproject.toml                  |

---

## Related Projects

- [Qwen1.5-0.5B][Qwen1.5-0.5B] — Base model
- [SmolLM2-360M-Instruct][SmolLM2-360M-Instruct] — Base model

---

## Contact

Bug reports and suggestions: **18200793117@163.com** | **tbnl_zldyd@outlook.com**

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jack66g/Growable-LLM&type=Date)](https://star-history.com/#jack66g/Growable-LLM&Date)

---

## License

This project is licensed under Apache License, Version 2.0.

[Qwen1.5-0.5B]: https://huggingface.co/Qwen/Qwen1.5-0.5B
[SmolLM2-360M-Instruct]: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
