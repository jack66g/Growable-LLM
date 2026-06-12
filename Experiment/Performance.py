# ============================================================
# Expansion Efficiency Benchmark
# Dynamic Parameter Growth Efficiency Suite
# ============================================================
#
# 目标:
# ------------------------------------------------------------
#
# 测试:
#
#   每增加参数
#   到底带来了多少性能收益
#
# ------------------------------------------------------------
#
# 核心问题:
#
# 1. 动态扩容是否真的有效？
# 2. 参数增长是否高效？
# 3. 是否出现 diminishing return？
# 4. 新容量是否真正吸收新知识？
#
# ------------------------------------------------------------
#
# 输出:
#
# 1. expansion_results.csv
# 2. scaling_law_curve.png
# 3. efficiency_curve.png
# 4. parameter_vs_loss.png
# 5. growth_gain_curve.png
#
# ------------------------------------------------------------
#
# 核心指标:
#
# Efficiency:
#
# ΔPerformance / ΔParams
#
# ------------------------------------------------------------
#
# 顶会风格:
#
# Scaling Law
# Parameter Efficiency
# Continual Expansion
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

from models import GrowableLLM, ModelConfig

# ============================================================
# Config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "outputs/expansion_efficiency"

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Expansion Schedule
# ============================================================

EXPANSION_STAGES = [
    0,
    64,
    128,
    256,
    512,
    768,
    1024,
]

TRAIN_STEPS = 60

BATCH_SIZE = 8

SEQ_LEN = 64

# ============================================================
# Synthetic Multi-Domain Data
# ============================================================

DOMAIN_NAMES = [
    "Medical",
    "Law",
    "Finance",
    "Code",
    "Math",
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
    """
    不同领域:
    - 不同 token distribution
    - 存在共享 latent structure
    """

    torch.manual_seed(domain_id * 9999)

    data = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.long
    )

    base_pattern = torch.arange(seq_len) % 13

    domain_shift = domain_id * 400

    for b in range(batch_size):

        noise = torch.randint(
            0,
            50,
            (seq_len,)
        )

        tokens = (
            base_pattern * 19
            + domain_shift
            + noise
        ) % vocab_size

        data[b] = tokens

    return data


# ============================================================
# Build Datasets
# ============================================================

def build_datasets(vocab_size):

    datasets = []

    for i in range(len(DOMAIN_NAMES)):

        data = generate_domain_data(
            vocab_size=vocab_size,
            domain_id=i
        )

        datasets.append(data.to(DEVICE))

    return datasets


# ============================================================
# Count Parameters
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


# ============================================================
# Evaluate
# ============================================================

@torch.no_grad()
def evaluate(model, datasets):

    model.eval()

    losses = []

    for data in datasets:

        _, loss = model(
            data,
            labels=data
        )

        losses.append(loss.item())

    return np.mean(losses)


# ============================================================
# Main Benchmark
# ============================================================

def run_expansion_efficiency_benchmark():

    print("\n===================================================")
    print("🚀 Expansion Efficiency Benchmark")
    print("===================================================\n")

    # ========================================================
    # Initial Model
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
    # Dataset
    # ========================================================

    datasets = build_datasets(config.vocab_size)

    # ========================================================
    # Result Storage
    # ========================================================

    results = []

    previous_loss = None

    previous_params = None

    # ========================================================
    # Expansion Loop
    # ========================================================

    for stage_id, expand_dim in enumerate(EXPANSION_STAGES):

        print("\n===================================================")
        print(f"📈 Expansion Stage {stage_id+1}")
        print(f"🧠 Expand Dim = {expand_dim}")
        print("===================================================\n")

        # ====================================================
        # Dynamic Expansion
        # ====================================================

        if expand_dim > 0:

            model.expand_model(
                extra_dim=expand_dim
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
        # Train Multi-Domain
        # ====================================================

        model.train()

        for step in tqdm(
            range(TRAIN_STEPS),
            desc=f"Training Stage {stage_id+1}"
        ):

            total_loss = 0

            optimizer.zero_grad()

            for domain_data in datasets:

                _, loss = model(
                    domain_data,
                    labels=domain_data
                )

                total_loss += loss

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            if step % 10 == 0:

                print(
                    f"Step {step:03d} "
                    f"| MultiDomain Loss = "
                    f"{total_loss.item():.4f}"
                )

        # ====================================================
        # Replay-Free Fusion
        # ====================================================

        fusion_optimizer = torch.optim.AdamW(
            filter(
                lambda p: p.requires_grad,
                model.parameters()
            ),
            lr=1e-4
        )

        model.defrag(
            fusion_optimizer,
            datasets[0]
        )

        # ====================================================
        # Evaluation
        # ====================================================

        avg_loss = evaluate(
            model,
            datasets
        )

        # ====================================================
        # Parameter Count
        # ====================================================

        param_count = count_parameters(model)

        # ====================================================
        # Compute Efficiency
        # ====================================================

        if previous_loss is not None:

            delta_perf = previous_loss - avg_loss

            delta_param = (
                param_count - previous_params
            )

            efficiency = (
                delta_perf / delta_param
            )

        else:

            delta_perf = 0

            delta_param = 0

            efficiency = 0

        # ====================================================
        # Save Result
        # ====================================================

        result = {
            "stage": stage_id,
            "expand_dim": expand_dim,
            "params": param_count,
            "avg_loss": avg_loss,
            "delta_perf": delta_perf,
            "delta_param": delta_param,
            "efficiency": efficiency,
        }

        results.append(result)

        print("\n📊 Stage Result")
        print(f"Params       : {param_count:,}")
        print(f"Avg Loss     : {avg_loss:.4f}")
        print(f"ΔPerformance : {delta_perf:.6f}")
        print(f"ΔParams      : {delta_param:,}")
        print(f"Efficiency   : {efficiency:.12f}")

        previous_loss = avg_loss

        previous_params = param_count

    # ========================================================
    # Convert DataFrame
    # ========================================================

    df = pd.DataFrame(results)

    csv_path = os.path.join(
        SAVE_DIR,
        "expansion_results.csv"
    )

    df.to_csv(csv_path, index=False)

    print(f"\n✅ Saved CSV: {csv_path}")

    # ========================================================
    # Plot 1: Scaling Law
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["params"],
        df["avg_loss"],
        marker='o',
        linewidth=3
    )

    plt.title(
        "Scaling Law Curve",
        fontsize=18
    )

    plt.xlabel(
        "Parameter Count",
        fontsize=14
    )

    plt.ylabel(
        "Average Loss",
        fontsize=14
    )

    plt.grid(True)

    scaling_path = os.path.join(
        SAVE_DIR,
        "scaling_law_curve.png"
    )

    plt.savefig(
        scaling_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Scaling Curve: {scaling_path}")

    # ========================================================
    # Plot 2: Efficiency Curve
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["params"],
        df["efficiency"],
        marker='o',
        linewidth=3
    )

    plt.title(
        "Expansion Efficiency Curve",
        fontsize=18
    )

    plt.xlabel(
        "Parameter Count",
        fontsize=14
    )

    plt.ylabel(
        "ΔPerformance / ΔParams",
        fontsize=14
    )

    plt.grid(True)

    efficiency_path = os.path.join(
        SAVE_DIR,
        "efficiency_curve.png"
    )

    plt.savefig(
        efficiency_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Efficiency Curve: {efficiency_path}")

    # ========================================================
    # Plot 3: Parameter vs Loss
    # ========================================================

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        x=df["params"],
        y=df["avg_loss"],
        s=150
    )

    plt.title(
        "Parameter vs Loss",
        fontsize=18
    )

    plt.xlabel(
        "Parameter Count",
        fontsize=14
    )

    plt.ylabel(
        "Average Loss",
        fontsize=14
    )

    plt.grid(True)

    param_loss_path = os.path.join(
        SAVE_DIR,
        "parameter_vs_loss.png"
    )

    plt.savefig(
        param_loss_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Param-Loss Plot: {param_loss_path}")

    # ========================================================
    # Plot 4: Growth Gain Curve
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.bar(
        range(len(df)),
        df["delta_perf"]
    )

    plt.title(
        "Performance Gain Per Expansion",
        fontsize=18
    )

    plt.xlabel(
        "Expansion Stage",
        fontsize=14
    )

    plt.ylabel(
        "ΔPerformance",
        fontsize=14
    )

    plt.grid(True)

    gain_path = os.path.join(
        SAVE_DIR,
        "growth_gain_curve.png"
    )

    plt.savefig(
        gain_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Gain Curve: {gain_path}")

    # ========================================================
    # Final Statistics
    # ========================================================

    best_efficiency = df["efficiency"].max()

    best_stage = df.iloc[
        df["efficiency"].idxmax()
    ]

    print("\n===================================================")
    print("📊 FINAL STATISTICS")
    print("===================================================\n")

    print(
        f"Best Efficiency : "
        f"{best_efficiency:.12f}"
    )

    print(
        f"Best Expand Dim : "
        f"{best_stage['expand_dim']}"
    )

    print(
        f"Final Params    : "
        f"{df.iloc[-1]['params']:,}"
    )

    print(
        f"Final Avg Loss  : "
        f"{df.iloc[-1]['avg_loss']:.4f}"
    )

    print("\n===================================================")
    print("🎉 Benchmark Finished")
    print("===================================================\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_expansion_efficiency_benchmark()