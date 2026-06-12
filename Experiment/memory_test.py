# ============================================================
# Continual Domain Benchmark (Top-Tier Conference Style)
# GrowableLLM Continual Learning Evaluation Suite
# ============================================================
#
# 功能:
# 1. 多领域持续学习
# 2. 动态扩容
# 3. 无回放训练
# 4. BWT 指标
# 5. FWT 指标
# 6. Avg Accuracy
# 7. Forgetting Curve
# 8. Interference Heatmap
# 9. Top-tier 风格可视化
#
# 输出:
# - continual_results.csv
# - forgetting_curve.png
# - interference_heatmap.png
# - domain_accuracy_curve.png
#
# ============================================================

import os
import math
import random
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from models import GrowableLLM, ModelConfig

# ============================================================
# Global Config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_DOMAINS = 10
TRAIN_STEPS = 50
BATCH_SIZE = 8
SEQ_LEN = 64

EXPAND_PER_STAGE = 64

SAVE_DIR = "outputs/continual_benchmark"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Domain Definitions
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
# Synthetic Domain Data
# ============================================================

def generate_domain_data(
    vocab_size,
    domain_id,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN
):
    """
    构造:
    - 每个领域有不同 token 分布
    - 同时保留部分共享模式
    """

    torch.manual_seed(domain_id * 9999)

    data = torch.zeros((batch_size, seq_len), dtype=torch.long)

    base_pattern = torch.arange(seq_len) % 7

    domain_shift = domain_id * 300

    for b in range(batch_size):

        noise = torch.randint(0, 20, (seq_len,))

        tokens = (
            base_pattern * 13
            + domain_shift
            + noise
        ) % vocab_size

        data[b] = tokens

    return data


# ============================================================
# Build Dataset
# ============================================================

def build_all_datasets(vocab_size):

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
# Compute Metrics
# ============================================================

def compute_bwt(acc_matrix):

    """
    BWT = average(
        final_performance - original_performance
    )
    """

    T = len(acc_matrix)

    vals = []

    for i in range(T - 1):

        original = acc_matrix[i][i]
        final = acc_matrix[T - 1][i]

        vals.append(final - original)

    return np.mean(vals)


def compute_avg_accuracy(acc_matrix):

    final_row = acc_matrix[-1]

    return np.mean(final_row)


def compute_fwt(acc_matrix):

    """
    简化版 FWT
    """

    vals = []

    for i in range(1, len(acc_matrix)):

        before_learning = acc_matrix[i - 1][i]

        vals.append(before_learning)

    return np.mean(vals)


# ============================================================
# Main Benchmark
# ============================================================

def run_continual_benchmark():

    print("\n===================================================")
    print("🚀 Continual Domain Benchmark Starting")
    print("===================================================\n")

    # ========================================================
    # Model
    # ========================================================

    config = ModelConfig(
        vocab_size=10000,
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

    datasets = build_all_datasets(config.vocab_size)

    # ========================================================
    # Storage
    # ========================================================

    acc_matrix = []

    forgetting_curve = []

    # ========================================================
    # Continual Learning Loop
    # ========================================================

    for stage in tqdm(range(NUM_DOMAINS), desc="Continual Learning"):

        print(f"\n===================================================")
        print(f"📚 Learning Domain {stage+1}: {DOMAIN_NAMES[stage]}")
        print("===================================================\n")

        # ====================================================
        # Dynamic Expansion
        # ====================================================

        if stage > 0:

            model.expand_model(extra_dim=EXPAND_PER_STAGE)

        # ====================================================
        # Optimizer
        # ====================================================

        trainable_params = filter(
            lambda p: p.requires_grad,
            model.parameters()
        )

        optimizer = torch.optim.AdamW(
            trainable_params,
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
                    f"Step {step:03d} | Loss = {loss.item():.4f}"
                )

        # ====================================================
        # Defrag Fusion
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
        # Evaluate ALL learned domains
        # ====================================================

        model.eval()

        current_results = []

        with torch.no_grad():

            for eval_domain in range(NUM_DOMAINS):

                eval_data = datasets[eval_domain]

                eval_loss = evaluate(model, eval_data)

                current_results.append(eval_loss)

                print(
                    f"Eval Domain {eval_domain+1:02d} "
                    f"({DOMAIN_NAMES[eval_domain]}) "
                    f"Loss = {eval_loss:.4f}"
                )

        acc_matrix.append(current_results)

        # ====================================================
        # Forgetting
        # ====================================================

        if stage > 0:

            forgetting = []

            for old_domain in range(stage):

                original = acc_matrix[old_domain][old_domain]

                now = current_results[old_domain]

                forgetting.append(now - original)

            forgetting_curve.append(np.mean(forgetting))

        else:
            forgetting_curve.append(0)

    # ========================================================
    # Convert to numpy
    # ========================================================

    acc_matrix = np.array(acc_matrix)

    # ========================================================
    # Metrics
    # ========================================================

    BWT = compute_bwt(acc_matrix)

    FWT = compute_fwt(acc_matrix)

    AVG_ACC = compute_avg_accuracy(acc_matrix)

    print("\n===================================================")
    print("📊 FINAL METRICS")
    print("===================================================")

    print(f"BWT       : {BWT:.4f}")
    print(f"FWT       : {FWT:.4f}")
    print(f"AVG ACC   : {AVG_ACC:.4f}")

    # ========================================================
    # Save CSV
    # ========================================================

    df = pd.DataFrame(
        acc_matrix,
        columns=DOMAIN_NAMES,
        index=[
            f"After_{d+1}"
            for d in range(NUM_DOMAINS)
        ]
    )

    csv_path = os.path.join(
        SAVE_DIR,
        "continual_results.csv"
    )

    df.to_csv(csv_path)

    print(f"\n✅ CSV saved: {csv_path}")

    # ========================================================
    # Plot 1: Forgetting Curve
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, NUM_DOMAINS + 1),
        forgetting_curve,
        marker='o',
        linewidth=3
    )

    plt.title(
        "Forgetting Curve",
        fontsize=18
    )

    plt.xlabel(
        "Number of Learned Domains",
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

    print(f"✅ Forgetting curve saved: {forgetting_path}")

    # ========================================================
    # Plot 2: Interference Heatmap
    # ========================================================

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        acc_matrix,
        annot=True,
        cmap="turbo",
        xticklabels=DOMAIN_NAMES,
        yticklabels=[
            f"After_{i+1}"
            for i in range(NUM_DOMAINS)
        ]
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

    print(f"✅ Heatmap saved: {heatmap_path}")

    # ========================================================
    # Plot 3: Domain Curves
    # ========================================================

    plt.figure(figsize=(14, 8))

    colors = sns.color_palette(
        "husl",
        NUM_DOMAINS
    )

    for domain_id in range(NUM_DOMAINS):

        curve = acc_matrix[:, domain_id]

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

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left'
    )

    plt.grid(True)

    domain_curve_path = os.path.join(
        SAVE_DIR,
        "domain_accuracy_curve.png"
    )

    plt.savefig(
        domain_curve_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Domain curves saved: {domain_curve_path}")

    print("\n===================================================")
    print("🎉 Benchmark Finished")
    print("===================================================\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_continual_benchmark()