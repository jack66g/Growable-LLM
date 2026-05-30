# ============================================================
# TRUE NO-LOCK BASELINE
# Catastrophic Forgetting Benchmark
# ============================================================
#
# 核心目标:
# ------------------------------------------------------------
#
# 真正验证:
#
#   不使用 Hook Lock
#   不使用 Routing
#   不使用 Replay
#   不使用 Protection
#
# 是否会发生:
#
#   catastrophic forgetting
#
# ------------------------------------------------------------
#
# 这是:
#
#   最纯粹的 continual learning baseline
#
# ------------------------------------------------------------
#
# 与你的方法对比:
#
#   Ours:
#       Hook Lock + Expansion + Routing
#
#   Baseline:
#       Full Finetune + Expansion ONLY
#
# ------------------------------------------------------------
#
# 这个实验非常关键。
#
# 因为:
#
#   如果:
#
#   NoLock:
#       旧任务崩塌
#
#   HookLock:
#       旧任务稳定
#
# 那么:
#
#   直接证明:
#
#       Lock Mechanism 有效
#
# ============================================================

import os
import copy
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from model import GrowableLLM, ModelConfig

# ============================================================
# Config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "true_nolock_baseline"

os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_STEPS = 120

BATCH_SIZE = 8

SEQ_LEN = 64

EXPAND_DIM = 64

LR = 1e-3

# ============================================================
# Domains
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
]

NUM_DOMAINS = len(DOMAIN_NAMES)

# ============================================================
# Generate Synthetic Domain Data
# ============================================================

def generate_domain_data(
    vocab_size,
    domain_id,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN
):

    torch.manual_seed(domain_id * 7777)

    x = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.long
    )

    base_pattern = torch.arange(seq_len)

    domain_shift = domain_id * 1000

    for b in range(batch_size):

        noise = torch.randint(
            0,
            50,
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
# Build Dataset
# ============================================================

def build_datasets(vocab_size):

    datasets = []

    for i in range(NUM_DOMAINS):

        data = generate_domain_data(
            vocab_size=vocab_size,
            domain_id=i
        )

        datasets.append(data)

    return datasets

# ============================================================
# Evaluate
# ============================================================

@torch.no_grad()
def evaluate(model, x):

    model.eval()

    _, loss = model(
        x,
        labels=x
    )

    return loss.item()

# ============================================================
# Count Parameters
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )

# ============================================================
# Save State
# ============================================================

def save_model_state(model):

    return {

        k: v.detach().cpu().clone()

        for k, v in model.state_dict().items()
    }

# ============================================================
# Compute OLD MEMORY Drift
# ============================================================

def compute_old_memory_drift(
    old_state,
    new_state
):

    total_drift = 0.0

    total_count = 0

    for k in old_state.keys():

        if k not in new_state:
            continue

        old_p = old_state[k]
        new_p = new_state[k]

        if not torch.is_tensor(old_p):
            continue

        if not torch.is_tensor(new_p):
            continue

        # ====================================================
        # Compare ONLY shared region
        # ====================================================

        if old_p.shape != new_p.shape:

            min_shape = []

            for d_old, d_new in zip(
                old_p.shape,
                new_p.shape
            ):

                min_shape.append(
                    min(d_old, d_new)
                )

            # =================================================
            # Slice shared region
            # =================================================

            if len(min_shape) == 1:

                old_p = old_p[
                    :min_shape[0]
                ]

                new_p = new_p[
                    :min_shape[0]
                ]

            elif len(min_shape) == 2:

                old_p = old_p[
                    :min_shape[0],
                    :min_shape[1]
                ]

                new_p = new_p[
                    :min_shape[0],
                    :min_shape[1]
                ]

            elif len(min_shape) == 3:

                old_p = old_p[
                    :min_shape[0],
                    :min_shape[1],
                    :min_shape[2]
                ]

                new_p = new_p[
                    :min_shape[0],
                    :min_shape[1],
                    :min_shape[2]
                ]

        # ====================================================
        # Drift
        # ====================================================

        drift = (
            old_p.float()
            - new_p.float()
        ).abs().mean().item()

        total_drift += drift

        total_count += 1

    if total_count == 0:
        return 0.0

    return total_drift / total_count

# ============================================================
# Main Benchmark
# ============================================================

def run_true_nolock_baseline():

    print("\n===================================================")
    print("🚨 TRUE NO-LOCK BASELINE")
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
    # Dataset
    # ========================================================

    datasets = build_datasets(
        config.vocab_size
    )

    # ========================================================
    # Metrics
    # ========================================================

    all_task_losses = []

    forgetting_scores = []

    drift_history = []

    results = []

    # ========================================================
    # Initial State
    # ========================================================

    previous_state = save_model_state(model)

    # ========================================================
    # Continual Learning
    # ========================================================

    for stage in range(NUM_DOMAINS):

        print("\n===================================================")
        print(f"🧠 Learning {DOMAIN_NAMES[stage]}")
        print("===================================================\n")

        # ====================================================
        # EXPANSION ONLY
        # ====================================================

        if stage > 0:

            print(
                f"💥 Expand +{EXPAND_DIM}"
            )

            model.expand_model(
                extra_dim=EXPAND_DIM
            )

        # ====================================================
        # NO LOCK
        # ====================================================
        #
        # 所有参数:
        #
        # FULL UPDATE
        #
        # ====================================================

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR
        )

        # ====================================================
        # ONLY TRAIN CURRENT TASK
        # ====================================================
        #
        # NO REPLAY
        # NO MEMORY
        #
        # 这是最关键的地方
        #
        # ====================================================

        current_x = datasets[stage]

        model.train()

        for step in tqdm(
            range(TRAIN_STEPS),
            desc=f"Training {DOMAIN_NAMES[stage]}"
        ):

            optimizer.zero_grad()

            _, loss = model(
                current_x,
                labels=current_x
            )

            loss.backward()

            optimizer.step()

            if step % 10 == 0:

                print(
                    f"Step {step:03d} "
                    f"| Loss={loss.item():.6f}"
                )

        # ====================================================
        # Evaluate ALL TASKS
        # ====================================================

        print("\n📊 Evaluating ALL Previous Tasks")

        current_task_losses = []

        for eval_id in range(NUM_DOMAINS):

            eval_x = datasets[eval_id]

            eval_loss = evaluate(
                model,
                eval_x
            )

            current_task_losses.append(
                eval_loss
            )

            print(
                f"{DOMAIN_NAMES[eval_id]} "
                f"Loss = {eval_loss:.6f}"
            )

        all_task_losses.append(
            current_task_losses
        )

        # ====================================================
        # Compute Forgetting
        # ====================================================

        current_forgetting = []

        for old_task in range(stage):

            # old best performance
            old_best = all_task_losses[
                old_task
            ][old_task]

            # current performance
            current_perf = current_task_losses[
                old_task
            ]

            forgetting = (
                current_perf
                - old_best
            )

            current_forgetting.append(
                forgetting
            )

        if len(current_forgetting) > 0:

            avg_forgetting = np.mean(
                current_forgetting
            )

        else:

            avg_forgetting = 0.0

        forgetting_scores.append(
            avg_forgetting
        )

        print(
            f"\n⚠ Avg Forgetting = "
            f"{avg_forgetting:.6f}"
        )

        # ====================================================
        # Compute OLD MEMORY Drift
        # ====================================================

        current_state = save_model_state(model)

        drift = compute_old_memory_drift(
            previous_state,
            current_state
        )

        drift_history.append(drift)

        previous_state = current_state

        print(
            f"⚠ Old Memory Drift = "
            f"{drift:.6f}"
        )

        # ====================================================
        # Save Result
        # ====================================================

        result = {

            "stage": stage,

            "domain": DOMAIN_NAMES[stage],

            "params": count_parameters(model),

            "avg_forgetting": avg_forgetting,

            "old_memory_drift": drift,

            "avg_loss": np.mean(
                current_task_losses
            ),
        }

        results.append(result)

    # ========================================================
    # Save CSV
    # ========================================================

    df = pd.DataFrame(results)

    csv_path = os.path.join(
        SAVE_DIR,
        "true_nolock_results.csv"
    )

    df.to_csv(csv_path, index=False)

    print(f"\n✅ Saved CSV: {csv_path}")

    # ========================================================
    # Convert Loss Matrix
    # ========================================================

    loss_matrix = np.array(
        all_task_losses
    )

    # ========================================================
    # Plot 1: Forgetting Curve
    # ========================================================

    plt.figure(figsize=(10,6))

    plt.plot(
        forgetting_scores,
        marker='o',
        linewidth=3
    )

    plt.title(
        "Catastrophic Forgetting Curve",
        fontsize=18
    )

    plt.xlabel(
        "Training Stage",
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

    print(f"✅ Saved Forgetting Curve")

    # ========================================================
    # Plot 2: Task Loss Matrix
    # ========================================================

    plt.figure(figsize=(12,8))

    sns.heatmap(
        loss_matrix,
        annot=True,
        cmap="magma",
        xticklabels=DOMAIN_NAMES,
        yticklabels=[
            f"After_{i+1}"
            for i in range(NUM_DOMAINS)
        ]
    )

    plt.title(
        "Task Performance Matrix",
        fontsize=18
    )

    matrix_path = os.path.join(
        SAVE_DIR,
        "task_loss_matrix.png"
    )

    plt.savefig(
        matrix_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Task Matrix")

    # ========================================================
    # Plot 3: Old Memory Drift
    # ========================================================

    plt.figure(figsize=(10,6))

    plt.plot(
        drift_history,
        marker='o',
        linewidth=3
    )

    plt.title(
        "Old Memory Parameter Drift",
        fontsize=18
    )

    plt.xlabel(
        "Training Stage",
        fontsize=14
    )

    plt.ylabel(
        "Drift",
        fontsize=14
    )

    plt.grid(True)

    drift_path = os.path.join(
        SAVE_DIR,
        "old_memory_drift.png"
    )

    plt.savefig(
        drift_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Drift Curve")

    # ========================================================
    # Plot 4: Catastrophic Forgetting Heatmap
    # ========================================================

    forgetting_matrix = np.zeros(
        (
            NUM_DOMAINS,
            NUM_DOMAINS
        )
    )

    for stage in range(NUM_DOMAINS):

        for old_task in range(stage):

            old_best = all_task_losses[
                old_task
            ][old_task]

            current_perf = all_task_losses[
                stage
            ][old_task]

            forgetting = (
                current_perf
                - old_best
            )

            forgetting_matrix[
                old_task,
                stage
            ] = forgetting

    plt.figure(figsize=(12,8))

    sns.heatmap(
        forgetting_matrix,
        annot=True,
        cmap="viridis",
        xticklabels=DOMAIN_NAMES,
        yticklabels=DOMAIN_NAMES
    )

    plt.title(
        "Catastrophic Forgetting Matrix",
        fontsize=18
    )

    forgetting_heatmap_path = os.path.join(
        SAVE_DIR,
        "forgetting_matrix.png"
    )

    plt.savefig(
        forgetting_heatmap_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Forgetting Matrix")

    # ========================================================
    # Final Summary
    # ========================================================

    avg_forgetting = np.mean(
        forgetting_scores
    )

    avg_drift = np.mean(
        drift_history
    )

    print("\n===================================================")
    print("📊 FINAL SUMMARY")
    print("===================================================\n")

    print(
        f"Average Forgetting : "
        f"{avg_forgetting:.6f}"
    )

    print(
        f"Average Drift      : "
        f"{avg_drift:.6f}"
    )

    print("\n===================================================")
    print("🚨 TRUE NO-LOCK Finished")
    print("===================================================\n")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_true_nolock_baseline()