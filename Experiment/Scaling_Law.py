# ============================================================
# Scaling Law Benchmark
# Neural Continual Scaling Research Suite
# ============================================================
#
# 目标:
# ------------------------------------------------------------
#
# 测试:
#
#   动态扩容后
#   是否满足 Scaling Law
#
# ------------------------------------------------------------
#
# 核心问题:
#
# 1. Loss 是否随参数增长而下降？
# 2. 是否存在 power-law scaling？
# 3. 是否存在 capacity phase transition？
# 4. 是否出现 saturation region？
# 5. 是否出现 routing bottleneck？
#
# ------------------------------------------------------------
#
# 输出:
#
# 1. scaling_results.csv
# 2. scaling_curve.png
# 3. log_scaling_curve.png
# 4. power_law_fit.png
# 5. efficiency_decay_curve.png
# 6. phase_transition_curve.png
#
# ------------------------------------------------------------
#
# 核心指标:
#
# Scaling Exponent
#
# ------------------------------------------------------------
#
# Scaling Law:
#
# L(N)=aN^{-α}+b
#
# ------------------------------------------------------------
#
# 其中:
#
# N:
#   参数量
#
# α:
#   scaling exponent
#
# b:
#   irreducible loss
#
# ============================================================

import os
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from scipy.optimize import curve_fit

from model import GrowableLLM, ModelConfig

# ============================================================
# Config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "scaling_law_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Scaling Schedule
# ============================================================

EXPANSION_SCHEDULE = [
    0,
    32,
    64,
    128,
    256,
    512,
    768,
    1024,
    1536,
]

TRAIN_STEPS = 80

BATCH_SIZE = 8

SEQ_LEN = 64

# ============================================================
# Domains
# ============================================================

DOMAIN_NAMES = [
    "Medical",
    "Law",
    "Finance",
    "Code",
    "Math",
    "Reasoning",
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

    data = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.long
    )

    base_pattern = torch.arange(seq_len) % 17

    domain_shift = domain_id * 700

    for b in range(batch_size):

        noise = torch.randint(
            0,
            50,
            (seq_len,)
        )

        tokens = (
            base_pattern * 23
            + domain_shift
            + noise
        ) % vocab_size

        data[b] = tokens

    return data


# ============================================================
# Build Dataset
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
# Power Law
# ============================================================

def power_law(
    N,
    a,
    alpha,
    b
):
    return a * (N ** (-alpha)) + b


# ============================================================
# Main Benchmark
# ============================================================

def run_scaling_law_benchmark():

    print("\n===================================================")
    print("🚀 Scaling Law Benchmark")
    print("===================================================\n")

    # ========================================================
    # Base Model
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
    # Storage
    # ========================================================

    results = []

    previous_loss = None

    # ========================================================
    # Scaling Loop
    # ========================================================

    for stage, expand_dim in enumerate(
        EXPANSION_SCHEDULE
    ):

        print("\n===================================================")
        print(f"📈 Scaling Stage {stage+1}")
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
        # Multi-Domain Training
        # ====================================================

        model.train()

        for step in tqdm(
            range(TRAIN_STEPS),
            desc=f"Training Stage {stage+1}"
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
                    f"| Multi Loss = "
                    f"{total_loss.item():.4f}"
                )

        # ====================================================
        # Replay-Free Defrag
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
        # Evaluate
        # ====================================================

        avg_loss = evaluate(
            model,
            datasets
        )

        # ====================================================
        # Parameter Count
        # ====================================================

        param_count = count_parameters(
            model
        )

        # ====================================================
        # Delta
        # ====================================================

        if previous_loss is not None:

            delta_loss = (
                previous_loss - avg_loss
            )

        else:

            delta_loss = 0

        previous_loss = avg_loss

        # ====================================================
        # Save
        # ====================================================

        result = {
            "stage": stage,
            "expand_dim": expand_dim,
            "params": param_count,
            "avg_loss": avg_loss,
            "delta_loss": delta_loss,
        }

        results.append(result)

        print("\n📊 Stage Result")
        print(f"Params     : {param_count:,}")
        print(f"Avg Loss   : {avg_loss:.6f}")
        print(f"ΔLoss      : {delta_loss:.6f}")

    # ========================================================
    # DataFrame
    # ========================================================

    df = pd.DataFrame(results)

    csv_path = os.path.join(
        SAVE_DIR,
        "scaling_results.csv"
    )

    df.to_csv(csv_path, index=False)

    print(f"\n✅ Saved CSV: {csv_path}")

    # ========================================================
    # Fit Power Law
    # ========================================================

    x_data = np.array(
        df["params"]
    )

    y_data = np.array(
        df["avg_loss"]
    )

    popt, _ = curve_fit(
        power_law,
        x_data,
        y_data,
        maxfev=10000
    )

    a_fit, alpha_fit, b_fit = popt

    print("\n===================================================")
    print("📊 POWER LAW FIT")
    print("===================================================\n")

    print(f"a       = {a_fit:.6f}")
    print(f"alpha   = {alpha_fit:.6f}")
    print(f"b       = {b_fit:.6f}")

    # ========================================================
    # Predict
    # ========================================================

    y_fit = power_law(
        x_data,
        *popt
    )

    # ========================================================
    # Plot 1: Scaling Curve
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_data,
        y_data,
        marker='o',
        linewidth=3,
        label="Observed"
    )

    plt.title(
        "Scaling Law Curve",
        fontsize=18
    )

    plt.xlabel(
        "Parameters",
        fontsize=14
    )

    plt.ylabel(
        "Average Loss",
        fontsize=14
    )

    plt.grid(True)

    scaling_path = os.path.join(
        SAVE_DIR,
        "scaling_curve.png"
    )

    plt.savefig(
        scaling_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Scaling Curve: {scaling_path}")

    # ========================================================
    # Plot 2: Log Scaling Curve
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.loglog(
        x_data,
        y_data,
        marker='o',
        linewidth=3
    )

    plt.title(
        "Log-Log Scaling Curve",
        fontsize=18
    )

    plt.xlabel(
        "log(Parameters)",
        fontsize=14
    )

    plt.ylabel(
        "log(Loss)",
        fontsize=14
    )

    plt.grid(True)

    log_path = os.path.join(
        SAVE_DIR,
        "log_scaling_curve.png"
    )

    plt.savefig(
        log_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Log Curve: {log_path}")

    # ========================================================
    # Plot 3: Power Law Fit
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_data,
        y_data,
        'o',
        label="Observed"
    )

    plt.plot(
        x_data,
        y_fit,
        linewidth=3,
        label=f"Fit α={alpha_fit:.4f}"
    )

    plt.title(
        "Power Law Fit",
        fontsize=18
    )

    plt.xlabel(
        "Parameters",
        fontsize=14
    )

    plt.ylabel(
        "Loss",
        fontsize=14
    )

    plt.legend()

    plt.grid(True)

    fit_path = os.path.join(
        SAVE_DIR,
        "power_law_fit.png"
    )

    plt.savefig(
        fit_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Power Fit: {fit_path}")

    # ========================================================
    # Plot 4: Efficiency Decay
    # ========================================================

    efficiencies = []

    for i in range(1, len(df)):

        delta_loss = (
            df.iloc[i-1]["avg_loss"]
            - df.iloc[i]["avg_loss"]
        )

        delta_param = (
            df.iloc[i]["params"]
            - df.iloc[i-1]["params"]
        )

        efficiency = (
            delta_loss / delta_param
        )

        efficiencies.append(efficiency)

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(df)),
        efficiencies,
        marker='o',
        linewidth=3
    )

    plt.title(
        "Efficiency Decay Curve",
        fontsize=18
    )

    plt.xlabel(
        "Scaling Stage",
        fontsize=14
    )

    plt.ylabel(
        "ΔLoss / ΔParams",
        fontsize=14
    )

    plt.grid(True)

    efficiency_path = os.path.join(
        SAVE_DIR,
        "efficiency_decay_curve.png"
    )

    plt.savefig(
        efficiency_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Efficiency Curve: {efficiency_path}")

    # ========================================================
    # Plot 5: Phase Transition Curve
    # ========================================================

    curvature = []

    for i in range(1, len(df)-1):

        prev_loss = df.iloc[i-1]["avg_loss"]

        curr_loss = df.iloc[i]["avg_loss"]

        next_loss = df.iloc[i+1]["avg_loss"]

        second_derivative = (
            next_loss
            - 2 * curr_loss
            + prev_loss
        )

        curvature.append(second_derivative)

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(df)-1),
        curvature,
        marker='o',
        linewidth=3
    )

    plt.title(
        "Phase Transition Curve",
        fontsize=18
    )

    plt.xlabel(
        "Scaling Stage",
        fontsize=14
    )

    plt.ylabel(
        "Second Derivative",
        fontsize=14
    )

    plt.grid(True)

    phase_path = os.path.join(
        SAVE_DIR,
        "phase_transition_curve.png"
    )

    plt.savefig(
        phase_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Phase Curve: {phase_path}")

    # ========================================================
    # Final Summary
    # ========================================================

    print("\n===================================================")
    print("📊 FINAL SUMMARY")
    print("===================================================\n")

    print(
        f"Scaling Exponent α : "
        f"{alpha_fit:.6f}"
    )

    print(
        f"Irreducible Loss b : "
        f"{b_fit:.6f}"
    )

    print(
        f"Final Loss : "
        f"{df.iloc[-1]['avg_loss']:.6f}"
    )

    print(
        f"Final Params : "
        f"{df.iloc[-1]['params']:,}"
    )

    print("\n===================================================")
    print("🎉 Benchmark Finished")
    print("===================================================\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_scaling_law_benchmark()