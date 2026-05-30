# ============================================================
# Cross-Interference Benchmark
# Top-Tier Continual Learning Interference Suite
# ============================================================
#
# 目标:
# ------------------------------------------------------------
# 测试:
#
#   学习领域 B 后
#   是否污染领域 A
#
# ------------------------------------------------------------
#
# 输出:
#
# 1. interference_matrix.csv
# 2. interference_heatmap.png
# 3. forgetting_curve.png
# 4. domain_retention_curve.png
# 5. stability_curve.png
#
# ------------------------------------------------------------
#
# 核心指标:
#
# Interference(i,j):
#
#   学习 j 后
#   对 i 的损伤
#
# ------------------------------------------------------------
#
# 公式:
#
# I(i,j) = Loss_after_j(i) - Loss_after_i(i)
#
# ------------------------------------------------------------
#
# 解释:
#
# 若:
#
# I(i,j) > 0
#
# 表示:
#
# 新领域 j 污染了旧领域 i
#
# ============================================================

import os
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from model import GrowableLLM, ModelConfig

# ============================================================
# Config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_DOMAINS = 10

TRAIN_STEPS = 50

BATCH_SIZE = 8

SEQ_LEN = 64

EXPAND_DIM = 64

SAVE_DIR = "cross_interference_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Domain Names
# ============================================================

DOMAIN_NAMES = [
    "Medical",
    "Law",
    "Finance",
    "Code",
    "Math",
    "ChineseWiki",
    "EnglishWiki",
    "Reasoning",
    "LongContext",
    "Instruction",
]

# ============================================================
# Domain Data Generator
# ============================================================

def generate_domain_data(
    vocab_size,
    domain_id,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN
):
    """
    构造:
    - 每个领域有独立 token 分布
    - 同时存在共享规律
    - 强制制造 latent interference
    """

    torch.manual_seed(domain_id * 9999)

    data = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.long
    )

    # 公共规律
    base_pattern = torch.arange(seq_len) % 11

    # 领域偏移
    domain_shift = domain_id * 200

    for b in range(batch_size):

        local_noise = torch.randint(
            0,
            30,
            (seq_len,)
        )

        tokens = (
            base_pattern * 17
            + domain_shift
            + local_noise
        ) % vocab_size

        data[b] = tokens

    return data


# ============================================================
# Build Dataset
# ============================================================

def build_datasets(vocab_size):

    datasets = []

    for i in range(NUM_DOMAINS):

        data = generate_domain_data(
            vocab_size=vocab_size,
            domain_id=i
        )

        datasets.append(data.to(DEVICE))

    return datasets


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, data):

    model.eval()

    _, loss = model(data, labels=data)

    return loss.item()


# ============================================================
# Main Benchmark
# ============================================================

def run_cross_interference_benchmark():

    print("\n===================================================")
    print("🚀 Cross-Interference Benchmark")
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
        initial_ffn_dim=512,
        max_seq_len=2048,
    )

    model = GrowableLLM(config).to(DEVICE)

    # ========================================================
    # Dataset
    # ========================================================

    datasets = build_datasets(config.vocab_size)

    # ========================================================
    # Storage
    # ========================================================

    loss_matrix = []

    # ========================================================
    # Continual Learning
    # ========================================================

    for stage in tqdm(range(NUM_DOMAINS), desc="Growing Domains"):

        print("\n===================================================")
        print(f"📚 Learning Domain {stage+1}")
        print(f"🧠 {DOMAIN_NAMES[stage]}")
        print("===================================================\n")

        # ====================================================
        # Expand Model
        # ====================================================

        if stage > 0:

            model.expand_model(
                extra_dim=EXPAND_DIM
            )

        # ====================================================
        # Optimizer
        # ====================================================

        optimizer = torch.optim.AdamW(
            filter(
                lambda p: p.requires_grad,
                model.parameters()
            ),
            lr=1e-3,
            weight_decay=0.0
        )

        current_data = datasets[stage]

        # ====================================================
        # Train Current Domain
        # ====================================================

        model.train()

        for step in range(TRAIN_STEPS):

            optimizer.zero_grad()

            _, loss = model(
                current_data,
                labels=current_data
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            if step % 10 == 0:

                print(
                    f"Step {step:03d} "
                    f"| Loss = {loss.item():.4f}"
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
            current_data
        )

        # ====================================================
        # Evaluate ALL domains
        # ====================================================

        stage_losses = []

        with torch.no_grad():

            for eval_domain in range(NUM_DOMAINS):

                eval_data = datasets[eval_domain]

                eval_loss = evaluate(
                    model,
                    eval_data
                )

                stage_losses.append(eval_loss)

                print(
                    f"Eval Domain "
                    f"{eval_domain+1:02d} "
                    f"({DOMAIN_NAMES[eval_domain]}) "
                    f"Loss = {eval_loss:.4f}"
                )

        loss_matrix.append(stage_losses)

    # ========================================================
    # Convert
    # ========================================================

    loss_matrix = np.array(loss_matrix)

    # ========================================================
    # Build Interference Matrix
    # ========================================================

    """
    interference[i,j]

    表示:

    学习 j 后
    对 i 的损伤
    """

    interference_matrix = np.zeros(
        (NUM_DOMAINS, NUM_DOMAINS)
    )

    for i in range(NUM_DOMAINS):

        base_loss = loss_matrix[i][i]

        for j in range(i + 1, NUM_DOMAINS):

            later_loss = loss_matrix[j][i]

            interference = later_loss - base_loss

            interference_matrix[i][j] = interference

    # ========================================================
    # Save CSV
    # ========================================================

    df = pd.DataFrame(
        interference_matrix,
        index=DOMAIN_NAMES,
        columns=DOMAIN_NAMES
    )

    csv_path = os.path.join(
        SAVE_DIR,
        "interference_matrix.csv"
    )

    df.to_csv(csv_path)

    print(f"\n✅ Saved CSV: {csv_path}")

    # ========================================================
    # Plot 1: Interference Heatmap
    # ========================================================

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        interference_matrix,
        annot=True,
        cmap="turbo",
        xticklabels=DOMAIN_NAMES,
        yticklabels=DOMAIN_NAMES,
        linewidths=0.5
    )

    plt.title(
        "Cross-Domain Interference Matrix",
        fontsize=18
    )

    heatmap_path = os.path.join(
        SAVE_DIR,
        "interference_heatmap.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Heatmap: {heatmap_path}")

    # ========================================================
    # Plot 2: Forgetting Curve
    # ========================================================

    forgetting_curve = []

    for stage in range(NUM_DOMAINS):

        forgetting_vals = []

        for old_domain in range(stage):

            base = loss_matrix[old_domain][old_domain]

            current = loss_matrix[stage][old_domain]

            forgetting_vals.append(
                current - base
            )

        if len(forgetting_vals) > 0:

            forgetting_curve.append(
                np.mean(forgetting_vals)
            )

        else:
            forgetting_curve.append(0)

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, NUM_DOMAINS + 1),
        forgetting_curve,
        marker='o',
        linewidth=3
    )

    plt.title(
        "Average Forgetting Curve",
        fontsize=18
    )

    plt.xlabel(
        "Learning Stage",
        fontsize=14
    )

    plt.ylabel(
        "Average Forgetting",
        fontsize=14
    )

    plt.grid(True)

    forgetting_path = os.path.join(
        SAVE_DIR,
        "forgetting_curve.png"
    )

    plt.savefig(
        forgetting_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Forgetting Curve: {forgetting_path}")

    # ========================================================
    # Plot 3: Domain Retention Curves
    # ========================================================

    plt.figure(figsize=(14, 8))

    colors = sns.color_palette(
        "husl",
        NUM_DOMAINS
    )

    for domain_id in range(NUM_DOMAINS):

        curve = loss_matrix[:, domain_id]

        plt.plot(
            range(1, NUM_DOMAINS + 1),
            curve,
            marker='o',
            linewidth=2,
            label=DOMAIN_NAMES[domain_id],
            color=colors[domain_id]
        )

    plt.title(
        "Domain Retention Curves",
        fontsize=18
    )

    plt.xlabel(
        "Learning Stage",
        fontsize=14
    )

    plt.ylabel(
        "Validation Loss",
        fontsize=14
    )

    plt.grid(True)

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left'
    )

    retention_path = os.path.join(
        SAVE_DIR,
        "domain_retention_curve.png"
    )

    plt.savefig(
        retention_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Retention Curves: {retention_path}")

    # ========================================================
    # Plot 4: Stability Curve
    # ========================================================

    stability_curve = []

    for stage in range(NUM_DOMAINS):

        learned_losses = loss_matrix[
            stage,
            :stage+1
        ]

        stability = np.std(learned_losses)

        stability_curve.append(stability)

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, NUM_DOMAINS + 1),
        stability_curve,
        marker='o',
        linewidth=3
    )

    plt.title(
        "Model Stability Curve",
        fontsize=18
    )

    plt.xlabel(
        "Learning Stage",
        fontsize=14
    )

    plt.ylabel(
        "Loss Std",
        fontsize=14
    )

    plt.grid(True)

    stability_path = os.path.join(
        SAVE_DIR,
        "stability_curve.png"
    )

    plt.savefig(
        stability_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Stability Curve: {stability_path}")

    # ========================================================
    # Final Stats
    # ========================================================

    avg_interference = np.mean(
        interference_matrix[
            interference_matrix > 0
        ]
    )

    max_interference = np.max(
        interference_matrix
    )

    print("\n===================================================")
    print("📊 FINAL STATS")
    print("===================================================\n")

    print(f"Average Interference : {avg_interference:.4f}")
    print(f"Maximum Interference : {max_interference:.4f}")

    print("\n===================================================")
    print("🎉 Benchmark Finished")
    print("===================================================\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_cross_interference_benchmark()