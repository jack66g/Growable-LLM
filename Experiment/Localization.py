# ============================================================
# Knowledge Localization Benchmark
# Ring-Based Continual Transformer
# ============================================================
#
# 目标:
# ------------------------------------------------------------
#
# 验证:
#
#   不同领域知识
#   是否自动局部化到不同神经区域
#
# ------------------------------------------------------------
#
# 核心问题:
#
# 1. Medical / Code / Math
#    是否激活不同神经元？
#
# 2. 是否形成领域专属区域？
#
# 3. 是否形成 ring-like specialization？
#
# 4. overlap 是否随训练下降？
#
# ------------------------------------------------------------
#
# 理论:
#
# 如果 continual architecture 有效:
#
#   不同领域:
#
#       激活不同 neuron cluster
#
# ------------------------------------------------------------
#
# 输出:
#
# 1. localization_results.csv
# 2. domain_activation_heatmap.png
# 3. neuron_specialization.png
# 4. overlap_matrix.png
# 5. routing_entropy_curve.png
# 6. domain_localization_tsne.png
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
from sklearn.manifold import TSNE
from collections import defaultdict

from model import GrowableLLM, ModelConfig

# ============================================================
# Config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "knowledge_localization_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_STEPS = 60

BATCH_SIZE = 8

SEQ_LEN = 64

EXPAND_DIM = 64

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
# Generate Domain Data
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

    base_pattern = torch.arange(seq_len) % 23

    domain_shift = domain_id * 900

    for b in range(batch_size):

        noise = torch.randint(
            0,
            50,
            (seq_len,)
        )

        tokens = (
            base_pattern * 37
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
# Activation Storage
# ============================================================

activation_storage = {}

# ============================================================
# Hook Builder
# ============================================================

def build_hook(name):

    def hook(module, input, output):

        if isinstance(output, tuple):
            output = output[0]

        act = (
            output.detach()
            .abs()
            .mean(dim=(0,1))
        )

        activation_storage[name] = act.cpu()

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
# Extract Neuron Activations
# ============================================================

@torch.no_grad()
def extract_domain_signature(
    model,
    x
):

    activation_storage.clear()

    model.eval()

    _ = model(
        x,
        labels=x
    )

    signatures = []

    for module_name in sorted(
        activation_storage.keys()
    ):

        vec = activation_storage[
            module_name
        ]

        signatures.append(vec.numpy())

    signatures = np.concatenate(
        signatures,
        axis=0
    )

    return signatures

# ============================================================
# Overlap Score
# ============================================================

def overlap_score(
    sig1,
    sig2,
    topk_ratio=0.1
):

    k = int(len(sig1) * topk_ratio)

    idx1 = np.argsort(sig1)[-k:]

    idx2 = np.argsort(sig2)[-k:]

    overlap = len(
        set(idx1).intersection(set(idx2))
    )

    return overlap / k

# ============================================================
# Entropy
# ============================================================

def routing_entropy(signature):

    prob = signature / (
        signature.sum() + 1e-8
    )

    entropy = -np.sum(
        prob * np.log(prob + 1e-8)
    )

    return entropy

# ============================================================
# Main Benchmark
# ============================================================

def run_knowledge_localization_benchmark():

    print("\n===================================================")
    print("🚀 Knowledge Localization Benchmark")
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
    # Hooks
    # ========================================================

    hooks = register_hooks(model)

    # ========================================================
    # Dataset
    # ========================================================

    datasets = build_datasets(
        config.vocab_size
    )

    # ========================================================
    # Continual Learning
    # ========================================================

    for stage in range(NUM_DOMAINS):

        print("\n===================================================")
        print(f"🧠 Learning {DOMAIN_NAMES[stage]}")
        print("===================================================\n")

        # ====================================================
        # Expansion
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

        # ====================================================
        # Train
        # ====================================================

        x = datasets[stage]

        model.train()

        for step in tqdm(
            range(TRAIN_STEPS),
            desc=f"Training {DOMAIN_NAMES[stage]}"
        ):

            optimizer.zero_grad()

            _, loss = model(
                x,
                labels=x
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
                    f"| Loss = {loss.item():.6f}"
                )

    # ========================================================
    # Extract Domain Signatures
    # ========================================================

    print("\n===================================================")
    print("📊 Extracting Domain Signatures")
    print("===================================================\n")

    domain_signatures = {}

    for domain_id in range(NUM_DOMAINS):

        sig = extract_domain_signature(
            model,
            datasets[domain_id]
        )

        domain_signatures[
            DOMAIN_NAMES[domain_id]
        ] = sig

    # ========================================================
    # Save Activation Matrix
    # ========================================================

    activation_matrix = []

    for domain in DOMAIN_NAMES:

        activation_matrix.append(
            domain_signatures[domain]
        )

    activation_matrix = np.array(
        activation_matrix
    )

    df = pd.DataFrame(
        activation_matrix,
        index=DOMAIN_NAMES
    )

    csv_path = os.path.join(
        SAVE_DIR,
        "localization_results.csv"
    )

    df.to_csv(csv_path)

    print(f"✅ Saved CSV: {csv_path}")

    # ========================================================
    # Plot 1: Activation Heatmap
    # ========================================================

    plt.figure(figsize=(18, 8))

    sns.heatmap(
        activation_matrix,
        cmap="magma",
        yticklabels=DOMAIN_NAMES
    )

    plt.title(
        "Domain-Neuron Activation Heatmap",
        fontsize=20
    )

    plt.xlabel(
        "Neuron Index",
        fontsize=14
    )

    plt.ylabel(
        "Domain",
        fontsize=14
    )

    heatmap_path = os.path.join(
        SAVE_DIR,
        "domain_activation_heatmap.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Heatmap: {heatmap_path}")

    # ========================================================
    # Plot 2: Neuron Specialization
    # ========================================================

    specialization_scores = []

    for neuron_id in range(
        activation_matrix.shape[1]
    ):

        neuron_vec = activation_matrix[
            :,
            neuron_id
        ]

        specialization = (
            neuron_vec.max()
            /
            (neuron_vec.mean() + 1e-8)
        )

        specialization_scores.append(
            specialization
        )

    plt.figure(figsize=(14, 6))

    plt.plot(
        specialization_scores,
        linewidth=1.5
    )

    plt.title(
        "Neuron Specialization Scores",
        fontsize=18
    )

    plt.xlabel(
        "Neuron Index",
        fontsize=14
    )

    plt.ylabel(
        "Specialization",
        fontsize=14
    )

    plt.grid(True)

    spec_path = os.path.join(
        SAVE_DIR,
        "neuron_specialization.png"
    )

    plt.savefig(
        spec_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Specialization Curve: {spec_path}")

    # ========================================================
    # Plot 3: Overlap Matrix
    # ========================================================

    overlap_matrix = np.zeros(
        (
            NUM_DOMAINS,
            NUM_DOMAINS
        )
    )

    for i in range(NUM_DOMAINS):

        for j in range(NUM_DOMAINS):

            sig1 = activation_matrix[i]

            sig2 = activation_matrix[j]

            overlap = overlap_score(
                sig1,
                sig2
            )

            overlap_matrix[i, j] = overlap

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        overlap_matrix,
        annot=True,
        cmap="viridis",
        xticklabels=DOMAIN_NAMES,
        yticklabels=DOMAIN_NAMES
    )

    plt.title(
        "Domain Overlap Matrix",
        fontsize=18
    )

    overlap_path = os.path.join(
        SAVE_DIR,
        "overlap_matrix.png"
    )

    plt.savefig(
        overlap_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Overlap Matrix: {overlap_path}")

    # ========================================================
    # Plot 4: Routing Entropy
    # ========================================================

    entropies = []

    for domain in DOMAIN_NAMES:

        sig = domain_signatures[domain]

        entropy = routing_entropy(sig)

        entropies.append(entropy)

    plt.figure(figsize=(10, 6))

    plt.bar(
        DOMAIN_NAMES,
        entropies
    )

    plt.title(
        "Routing Entropy by Domain",
        fontsize=18
    )

    plt.ylabel(
        "Entropy",
        fontsize=14
    )

    plt.xticks(rotation=30)

    entropy_path = os.path.join(
        SAVE_DIR,
        "routing_entropy_curve.png"
    )

    plt.savefig(
        entropy_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved Entropy Curve: {entropy_path}")

    # ========================================================
    # Plot 5: t-SNE
    # ========================================================

    tsne = TSNE(
        n_components=2,
        perplexity=5,
        random_state=42
    )

    reduced = tsne.fit_transform(
        activation_matrix
    )

    plt.figure(figsize=(10, 8))

    for i, domain in enumerate(DOMAIN_NAMES):

        plt.scatter(
            reduced[i,0],
            reduced[i,1],
            s=200
        )

        plt.text(
            reduced[i,0],
            reduced[i,1],
            domain,
            fontsize=12
        )

    plt.title(
        "Domain Localization t-SNE",
        fontsize=18
    )

    tsne_path = os.path.join(
        SAVE_DIR,
        "domain_localization_tsne.png"
    )

    plt.savefig(
        tsne_path,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ Saved t-SNE: {tsne_path}")

    # ========================================================
    # Final Statistics
    # ========================================================

    avg_overlap = np.mean(
        overlap_matrix[
            np.triu_indices(NUM_DOMAINS, k=1)
        ]
    )

    avg_entropy = np.mean(entropies)

    avg_specialization = np.mean(
        specialization_scores
    )

    print("\n===================================================")
    print("📊 FINAL SUMMARY")
    print("===================================================\n")

    print(
        f"Average Overlap      : "
        f"{avg_overlap:.4f}"
    )

    print(
        f"Average Entropy      : "
        f"{avg_entropy:.4f}"
    )

    print(
        f"Average Specialization : "
        f"{avg_specialization:.4f}"
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

    run_knowledge_localization_benchmark()