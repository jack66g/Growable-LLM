# ============================================================
# Catastrophic Forgetting Stress Test
# Strong Conflict Continual Learning Benchmark
# ============================================================
#
# 目标:
# ------------------------------------------------------------
#
# 强制制造:
#
#   新知识 vs 旧知识
#
# 的:
#
#   完全冲突
#
# ------------------------------------------------------------
#
# 测试:
#
# 1. 是否发生 catastrophic forgetting
# 2. 是否形成 knowledge overwrite
# 3. 是否形成 dual-memory coexistence
# 4. 是否出现 representational collapse
#
# ------------------------------------------------------------
#
# 输出:
#
# 1. forgetting_results.csv
# 2. forgetting_curve.png
# 3. conflict_heatmap.png
# 4. knowledge_stability_curve.png
# 5. retention_curve.png
#
# ------------------------------------------------------------
#
# 核心指标:
#
# Forgetting Score
#
# ------------------------------------------------------------
#
# 定义:
#
# FS(i,j):
#
# 学习冲突任务 j 后
# 对任务 i 的遗忘程度
#
# ------------------------------------------------------------
#
# 公式:
#
# FS(i,j)=L_after_j(i)-L_after_i(i)
#
# ============================================================

import os
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

SAVE_DIR = "catastrophic_forgetting_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_STEPS = 60

BATCH_SIZE = 8

SEQ_LEN = 64

EXPAND_DIM = 64

# ============================================================
# Conflict Tasks
# ============================================================

"""
这里故意制造:

完全冲突知识

例如:

Task A:
    token 10 -> token 20

Task B:
    token 10 -> token 999

强制模型:
    出现覆盖冲突
"""

TASK_NAMES = [
    "Task_A",
    "Task_B",
    "Task_C",
    "Task_D",
    "Task_E",
]

# ============================================================
# Generate Conflict Dataset
# ============================================================

def generate_conflict_task(
    vocab_size,
    task_id,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN
):
    """
    每个 task:
    输入模式类似

    但 target 完全冲突
    """

    torch.manual_seed(task_id * 9999)

    x = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.long
    )

    y = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.long
    )

    base_pattern = torch.arange(seq_len) % 7

    for b in range(batch_size):

        noise = torch.randint(
            0,
            10,
            (seq_len,)
        )

        # 输入保持高度相似
        input_tokens = (
            base_pattern * 13
            + noise
        ) % vocab_size

        # 输出完全冲突
        target_shift = task_id * 500

        target_tokens = (
            base_pattern * 17
            + target_shift
            + noise
        ) % vocab_size

        x[b] = input_tokens

        y[b] = target_tokens

    return (
        x.to(DEVICE),
        y.to(DEVICE)
    )

# ============================================================
# Build Tasks
# ============================================================

def build_tasks(vocab_size):

    tasks = []

    for i in range(len(TASK_NAMES)):

        task = generate_conflict_task(
            vocab_size=vocab_size,
            task_id=i
        )

        tasks.append(task)

    return tasks

# ============================================================
# Evaluate
# ============================================================

@torch.no_grad()
def evaluate(model, x, y):

    model.eval()

    _, loss = model(
        x,
        labels=y
    )

    return loss.item()

# ============================================================
# Forgetting Score
# ============================================================

def compute_forgetting_score(loss_matrix):

    T = len(loss_matrix)

    forgetting_scores = []

    for i in range(T - 1):

        original = loss_matrix[i][i]

        final = loss_matrix[T - 1][i]

        forgetting = final - original

        forgetting_scores.append(forgetting)

    return np.mean(forgetting_scores)

# ============================================================
# Main Benchmark
# ============================================================

def run_catastrophic_forgetting_benchmark():

    print("\n===================================================")
    print("🚀 Catastrophic Forgetting Stress Test")
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
    # Tasks
    # ========================================================

    tasks = build_tasks(config.vocab_size)

    # ========================================================
    # Storage
    # ========================================================

    loss_matrix = []

    forgetting_curve = []

    # ========================================================
    # Continual Conflict Learning
    # ========================================================

    for stage in tqdm(
        range(len(TASK_NAMES)),
        desc="Conflict Learning"
    ):

        print("\n===================================================")
        print(f"⚠️ Learning Conflict Task {stage+1}")
        print(f"🧠 {TASK_NAMES[stage]}")
        print("===================================================\n")

        # ====================================================
        # Dynamic Expansion
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
            lr=1e-3
        )

        x_train, y_train = tasks[stage]

        # ====================================================
        # Train
        # ====================================================

        model.train()

        for step in range(TRAIN_STEPS):

            optimizer.zero_grad()

            _, loss = model(
                x_train,
                labels=y_train
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
            x_train
        )

        # ====================================================
        # Evaluate ALL tasks
        # ====================================================

        stage_losses = []

        with torch.no_grad():

            for eval_task in range(len(TASK_NAMES)):

                x_eval, y_eval = tasks[eval_task]

                eval_loss = evaluate(
                    model,
                    x_eval,
                    y_eval
                )

                stage_losses.append(eval_loss)

                print(
                    f"Eval Task "
                    f"{eval_task+1:02d} "
                    f"({TASK_NAMES[eval_task]}) "
                    f"Loss = {eval_loss:.4f}"
                )

        loss_matrix.append(stage_losses)

        # ====================================================
        # Forgetting Curve
        # ====================================================

        if stage > 0:

            forgetting_vals = []

            for old_task in range(stage):

                original = loss_matrix[
                    old_task
                ][old_task]

                current = stage_losses[
                    old_task
                ]

                forgetting_vals.append(
                    current - original
                )

            forgetting_curve.append(
                np.mean(forgetting_vals)
            )

        else:
            forgetting_curve.append(0)

    # ========================================================
    # Convert
    # ========================================================

    loss_matrix = np.array(loss_matrix)

    # ========================================================
    # Compute Forgetting Matrix
    # ========================================================

    forgetting_matrix = np.zeros(
        (
            len(TASK_NAMES),
            len(TASK_NAMES)
        )
    )

    for i in range(len(TASK_NAMES)):

        base_loss = loss_matrix[i][i]

        for j in range(i + 1, len(TASK_NAMES)):

            later_loss = loss_matrix[j][i]

            forgetting = later_loss - base_loss

            forgetting_matrix[i][j] = forgetting

    # ========================================================
    # Save CSV
    # ========================================================

    df = pd.DataFrame(
        forgetting_matrix,
        index=TASK_NAMES,
        columns=TASK_NAMES
    )

    csv_path = os.path.join(
        SAVE_DIR,
        "forgetting_results.csv"
    )

    df.to_csv(csv_path)

    print(f"\n✅ Saved CSV: {csv_path}")

    # ========================================================
    # Plot 1: Forgetting Heatmap
    # ========================================================

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        forgetting_matrix,
        annot=True,
        cmap="magma",
        xticklabels=TASK_NAMES,
        yticklabels=TASK_NAMES,
        linewidths=0.5
    )

    plt.title(
        "Catastrophic Forgetting Matrix",
        fontsize=18
    )

    heatmap_path = os.path.join(
        SAVE_DIR,
        "conflict_heatmap.png"
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

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(TASK_NAMES)+1),
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
    # Plot 3: Task Retention Curves
    # ========================================================

    plt.figure(figsize=(12, 7))

    colors = sns.color_palette(
        "husl",
        len(TASK_NAMES)
    )

    for task_id in range(len(TASK_NAMES)):

        curve = loss_matrix[:, task_id]

        plt.plot(
            range(1, len(TASK_NAMES)+1),
            curve,
            marker='o',
            linewidth=2,
            label=TASK_NAMES[task_id],
            color=colors[task_id]
        )

    plt.title(
        "Task Retention Curves",
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

    plt.legend()

    plt.grid(True)

    retention_path = os.path.join(
        SAVE_DIR,
        "retention_curve.png"
    )

    plt.savefig(
        retention_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Retention Curve: {retention_path}")

    # ========================================================
    # Plot 4: Stability Curve
    # ========================================================

    stability_curve = []

    for stage in range(len(TASK_NAMES)):

        learned = loss_matrix[
            stage,
            :stage+1
        ]

        stability = np.std(learned)

        stability_curve.append(stability)

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(TASK_NAMES)+1),
        stability_curve,
        marker='o',
        linewidth=3
    )

    plt.title(
        "Knowledge Stability Curve",
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
        "knowledge_stability_curve.png"
    )

    plt.savefig(
        stability_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Stability Curve: {stability_path}")

    # ========================================================
    # Final Metrics
    # ========================================================

    final_fs = compute_forgetting_score(
        loss_matrix
    )

    max_forgetting = np.max(
        forgetting_matrix
    )

    avg_forgetting = np.mean(
        forgetting_matrix[
            forgetting_matrix > 0
        ]
    )

    print("\n===================================================")
    print("📊 FINAL METRICS")
    print("===================================================\n")

    print(f"Forgetting Score : {final_fs:.4f}")
    print(f"Average Forgetting : {avg_forgetting:.4f}")
    print(f"Maximum Forgetting : {max_forgetting:.4f}")

    print("\n===================================================")
    print("🎉 Benchmark Finished")
    print("===================================================\n")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_catastrophic_forgetting_benchmark()