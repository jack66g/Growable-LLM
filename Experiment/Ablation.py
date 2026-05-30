# ============================================================
# Neural Age Ablation Benchmark
# Ring-Based Continual Transformer
# ============================================================
#
# 目标:
# ------------------------------------------------------------
#
# 验证:
#
#   神经元是否随年龄增长
#   自动形成长期稳定记忆
#
# ------------------------------------------------------------
#
# 核心假设:
#
# Old Neurons:
#   更稳定
#   更低梯度
#   更低覆盖率
#
# Young Neurons:
#   更高可塑性
#   更高梯度
#   更高更新频率
#
# ------------------------------------------------------------
#
# 核心问题:
#
# 1. Plasticity 是否随年龄下降？
# 2. 老神经是否更抗覆盖？
# 3. 模型是否形成长期记忆层？
# 4. 是否出现神经生命周期结构？
#
# ------------------------------------------------------------
#
# 输出:
#
# 1. neural_age_results.csv
# 2. gradient_vs_age.png
# 3. overwrite_vs_age.png
# 4. activation_vs_age.png
# 5. plasticity_decay_curve.png
# 6. neuron_age_heatmap.png
#
# ------------------------------------------------------------
#
# 核心指标:
#
# Plasticity(age)
#
# ------------------------------------------------------------
#
# 理论:
#
# Plasticity(age)=e^{-λ age}
#
# ============================================================

import os
import math
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from collections import defaultdict

from model import GrowableLLM, ModelConfig

# ============================================================
# Config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "neural_age_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_STEPS = 80

BATCH_SIZE = 8

SEQ_LEN = 64

EXPAND_DIM = 64

NUM_STAGES = 8

# ============================================================
# Domain Tasks
# ============================================================

DOMAIN_NAMES = [
    "Medical",
    "Law",
    "Finance",
    "Code",
    "Math",
    "Reasoning",
    "ChineseWiki",
    "LongContext",
]

# ============================================================
# Generate Domain Data
# ============================================================

def generate_domain_data(
    vocab_size,
    domain_id,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN
):

    torch.manual_seed(domain_id * 9999)

    x = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.long
    )

    base_pattern = torch.arange(seq_len) % 19

    domain_shift = domain_id * 800

    for b in range(batch_size):

        noise = torch.randint(
            0,
            40,
            (seq_len,)
        )

        tokens = (
            base_pattern * 31
            + domain_shift
            + noise
        ) % vocab_size

        x[b] = tokens

    return x.to(DEVICE)

# ============================================================
# Build Tasks
# ============================================================

def build_tasks(vocab_size):

    tasks = []

    for i in range(NUM_STAGES):

        data = generate_domain_data(
            vocab_size=vocab_size,
            domain_id=i
        )

        tasks.append(data)

    return tasks

# ============================================================
# Count Parameters
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )

# ============================================================
# Neuron Birth Registry
# ============================================================

class NeuronAgeTracker:

    def __init__(self):

        self.birth_stage = {}

    def register_model(self, model, stage):

        for name, param in model.named_parameters():

            if "ffn" not in name.lower():
                continue

            if len(param.shape) < 2:
                continue

            out_dim = param.shape[0]

            for neuron_id in range(out_dim):

                key = f"{name}_{neuron_id}"

                if key not in self.birth_stage:

                    self.birth_stage[key] = stage

    def get_age(self, key, current_stage):

        birth = self.birth_stage[key]

        return current_stage - birth

# ============================================================
# Statistics Collector
# ============================================================

class NeuralStatistics:

    def __init__(self):

        self.gradient_by_age = defaultdict(list)

        self.overwrite_by_age = defaultdict(list)

        self.activation_by_age = defaultdict(list)

    def add_gradient(self, age, value):

        self.gradient_by_age[age].append(value)

    def add_overwrite(self, age, value):

        self.overwrite_by_age[age].append(value)

    def add_activation(self, age, value):

        self.activation_by_age[age].append(value)

# ============================================================
# Activation Hook
# ============================================================

activation_storage = {}

def build_hook(name):

    def hook(module, input, output):

        if isinstance(output, tuple):
            output = output[0]

        activation_storage[name] = (
            output.detach()
            .abs()
            .mean()
            .item()
        )

    return hook

# ============================================================
# Register Hooks
# ============================================================

def register_hooks(model):

    hooks = []

    for name, module in model.named_modules():

        if "ffn" in name.lower():

            h = module.register_forward_hook(
                build_hook(name)
            )

            hooks.append(h)

    return hooks

# ============================================================
# Main Benchmark
# ============================================================

def run_neural_age_benchmark():

    print("\n===================================================")
    print("🚀 Neural Age Ablation Benchmark")
    print("===================================================\n")

    # ========================================================
    # Model
    # ========================================================

    config = ModelConfig(
        vocab_size=12000,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=8,
        initial_ffn_dim=256,
        max_seq_len=2048,
    )

    model = GrowableLLM(config).to(DEVICE)

    # ========================================================
    # Tasks
    # ========================================================

    tasks = build_tasks(
        config.vocab_size
    )

    # ========================================================
    # Hooks
    # ========================================================

    hooks = register_hooks(model)

    # ========================================================
    # Trackers
    # ========================================================

    age_tracker = NeuronAgeTracker()

    stats = NeuralStatistics()

    # ========================================================
    # Previous Parameter Snapshot
    # ========================================================

    previous_params = {}

    # ========================================================
    # Continual Learning Loop
    # ========================================================

    for stage in range(NUM_STAGES):

        print("\n===================================================")
        print(f"🧠 Stage {stage+1}")
        print(f"📚 Domain = {DOMAIN_NAMES[stage]}")
        print("===================================================\n")

        # ====================================================
        # Expansion
        # ====================================================

        if stage > 0:

            model.expand_model(
                extra_dim=EXPAND_DIM
            )

        # ====================================================
        # Register Neuron Birth
        # ====================================================

        age_tracker.register_model(
            model,
            stage
        )

        # ====================================================
        # Save Previous Params
        # ====================================================

        previous_params = {}

        for name, param in model.named_parameters():

            previous_params[name] = (
                param.detach()
                .clone()
            )

        # ====================================================
        # Optimizer
        # ====================================================

        optimizer = torch.optim.AdamW(
            filter(
                lambda p: p.requires_grad,
                model.parameters()
            ),
            lr=1e-3
        )

        # ====================================================
        # Train
        # ====================================================

        model.train()

        x = tasks[stage]

        for step in tqdm(
            range(TRAIN_STEPS),
            desc=f"Training Stage {stage+1}"
        ):

            optimizer.zero_grad()

            _, loss = model(
                x,
                labels=x
            )

            loss.backward()

            # ================================================
            # Collect Gradient Statistics
            # ================================================

            for name, param in model.named_parameters():

                if param.grad is None:
                    continue

                if "ffn" not in name.lower():
                    continue

                if len(param.shape) < 2:
                    continue

                grad = (
                    param.grad.detach()
                    .abs()
                    .mean(dim=1)
                )

                for neuron_id in range(
                    grad.shape[0]
                ):

                    key = (
                        f"{name}_{neuron_id}"
                    )

                    age = age_tracker.get_age(
                        key,
                        stage
                    )

                    grad_value = (
                        grad[neuron_id]
                        .item()
                    )

                    stats.add_gradient(
                        age,
                        grad_value
                    )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            # ================================================
            # Logging
            # ================================================

            if step % 10 == 0:

                print(
                    f"Step {step:03d} "
                    f"| Loss = {loss.item():.6f}"
                )

        # ====================================================
        # Overwrite Statistics
        # ====================================================

        for name, param in model.named_parameters():

            if "ffn" not in name.lower():
                continue

            if len(param.shape) < 2:
                continue

            prev = previous_params[name]

            diff = (
                param.detach() - prev
            ).abs().mean(dim=1)

            for neuron_id in range(
                diff.shape[0]
            ):

                key = (
                    f"{name}_{neuron_id}"
                )

                age = age_tracker.get_age(
                    key,
                    stage
                )

                overwrite = (
                    diff[neuron_id]
                    .item()
                )

                stats.add_overwrite(
                    age,
                    overwrite
                )

        # ====================================================
        # Activation Statistics
        # ====================================================

        model.eval()

        with torch.no_grad():

            _ = model(
                x,
                labels=x
            )

        for module_name, value in activation_storage.items():

            fake_age = stage

            stats.add_activation(
                fake_age,
                value
            )

    # ========================================================
    # Aggregate Statistics
    # ========================================================

    ages = sorted(
        stats.gradient_by_age.keys()
    )

    gradient_means = []

    overwrite_means = []

    activation_means = []

    plasticity_scores = []

    for age in ages:

        g = np.mean(
            stats.gradient_by_age[age]
        )

        o = np.mean(
            stats.overwrite_by_age[age]
        )

        a = np.mean(
            stats.activation_by_age[age]
        )

        gradient_means.append(g)

        overwrite_means.append(o)

        activation_means.append(a)

        plasticity = g / (o + 1e-8)

        plasticity_scores.append(plasticity)

    # ========================================================
    # Save CSV
    # ========================================================

    df = pd.DataFrame({

        "age": ages,

        "gradient_mean": gradient_means,

        "overwrite_mean": overwrite_means,

        "activation_mean": activation_means,

        "plasticity_score": plasticity_scores,
    })

    csv_path = os.path.join(
        SAVE_DIR,
        "neural_age_results.csv"
    )

    df.to_csv(csv_path, index=False)

    print(f"\n✅ Saved CSV: {csv_path}")

    # ========================================================
    # Plot 1: Gradient vs Age
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        ages,
        gradient_means,
        marker='o',
        linewidth=3
    )

    plt.title(
        "Gradient Magnitude vs Neuron Age",
        fontsize=18
    )

    plt.xlabel(
        "Neuron Age",
        fontsize=14
    )

    plt.ylabel(
        "Gradient Magnitude",
        fontsize=14
    )

    plt.grid(True)

    grad_path = os.path.join(
        SAVE_DIR,
        "gradient_vs_age.png"
    )

    plt.savefig(
        grad_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Gradient Curve: {grad_path}")

    # ========================================================
    # Plot 2: Overwrite vs Age
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        ages,
        overwrite_means,
        marker='o',
        linewidth=3
    )

    plt.title(
        "Overwrite Magnitude vs Neuron Age",
        fontsize=18
    )

    plt.xlabel(
        "Neuron Age",
        fontsize=14
    )

    plt.ylabel(
        "Parameter Change",
        fontsize=14
    )

    plt.grid(True)

    overwrite_path = os.path.join(
        SAVE_DIR,
        "overwrite_vs_age.png"
    )

    plt.savefig(
        overwrite_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Overwrite Curve: {overwrite_path}")

    # ========================================================
    # Plot 3: Activation vs Age
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        ages,
        activation_means,
        marker='o',
        linewidth=3
    )

    plt.title(
        "Activation vs Neuron Age",
        fontsize=18
    )

    plt.xlabel(
        "Neuron Age",
        fontsize=14
    )

    plt.ylabel(
        "Activation Strength",
        fontsize=14
    )

    plt.grid(True)

    activation_path = os.path.join(
        SAVE_DIR,
        "activation_vs_age.png"
    )

    plt.savefig(
        activation_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Activation Curve: {activation_path}")

    # ========================================================
    # Plot 4: Plasticity Decay
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        ages,
        plasticity_scores,
        marker='o',
        linewidth=3
    )

    plt.title(
        "Plasticity Decay Curve",
        fontsize=18
    )

    plt.xlabel(
        "Neuron Age",
        fontsize=14
    )

    plt.ylabel(
        "Plasticity Score",
        fontsize=14
    )

    plt.grid(True)

    plasticity_path = os.path.join(
        SAVE_DIR,
        "plasticity_decay_curve.png"
    )

    plt.savefig(
        plasticity_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Plasticity Curve: {plasticity_path}")

    # ========================================================
    # Plot 5: Heatmap
    # ========================================================

    heatmap_data = np.stack([
        gradient_means,
        overwrite_means,
        activation_means,
        plasticity_scores,
    ])

    plt.figure(figsize=(12, 5))

    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="magma",
        xticklabels=ages,
        yticklabels=[
            "Gradient",
            "Overwrite",
            "Activation",
            "Plasticity",
        ]
    )

    plt.title(
        "Neuron Age Statistics Heatmap",
        fontsize=18
    )

    heatmap_path = os.path.join(
        SAVE_DIR,
        "neuron_age_heatmap.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Heatmap: {heatmap_path}")

    # ========================================================
    # Final Summary
    # ========================================================

    print("\n===================================================")
    print("📊 FINAL SUMMARY")
    print("===================================================\n")

    for age in ages:

        idx = ages.index(age)

        print(
            f"Age={age} | "
            f"Grad={gradient_means[idx]:.6f} | "
            f"Overwrite={overwrite_means[idx]:.6f} | "
            f"Plasticity={plasticity_scores[idx]:.6f}"
        )

    print("\n===================================================")
    print("🎉 Benchmark Finished")
    print("===================================================\n")

    # ========================================================
    # Remove Hooks
    # ========================================================

    for h in hooks:
        h.remove()

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_neural_age_benchmark()