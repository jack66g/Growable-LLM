# ============================================================
# NEURAL RING HOOK LOCK
# FINAL STABLE VERSION
# ============================================================

import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from tqdm import tqdm

from models import GrowableLLM, ModelConfig

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "outputs/hooklock_final"

os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_STEPS = 150

BATCH_SIZE = 8

SEQ_LEN = 64

LR = 1e-3

EXPAND_DIM = 64

# ============================================================
# DOMAINS
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
# GLOBAL
# ============================================================

locked_masks = {}

neuron_age_map = {}

activation_storage = {}

# ============================================================
# DATA
# ============================================================

def generate_domain_batch(
    vocab_size,
    domain_id,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN
):

    x = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.long
    )

    domain_seed = (
        domain_id * 9999
        + random.randint(0, 9999)
    )

    torch.manual_seed(domain_seed)

    base_pattern = torch.arange(seq_len)

    domain_shift = domain_id * 1300

    for b in range(batch_size):

        noise = torch.randint(
            0,
            50,
            (seq_len,)
        )

        tokens = (
            base_pattern * (domain_id + 17)
            + domain_shift
            + noise
        ) % vocab_size

        x[b] = tokens

    return x.to(DEVICE)

# ============================================================
# EVAL
# ============================================================

@torch.no_grad()
def evaluate_domain(
    model,
    vocab_size,
    domain_id
):

    model.eval()

    losses = []

    for _ in range(5):

        x = generate_domain_batch(
            vocab_size,
            domain_id
        )

        _, loss = model(
            x,
            labels=x
        )

        losses.append(loss.item())

    return np.mean(losses)

# ============================================================
# PARAM COUNT
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )

# ============================================================
# SAVE MODEL STATE
# ============================================================

def save_model_state(model):

    return {

        k: v.detach().cpu().clone()

        for k, v in model.state_dict().items()
    }

# ============================================================
# DRIFT
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

        if "ffn" not in k.lower():
            continue

        old_p = old_state[k]

        new_p = new_state[k]

        if not torch.is_tensor(old_p):
            continue

        if not torch.is_tensor(new_p):
            continue

        if old_p.shape != new_p.shape:

            min_shape = []

            for d1, d2 in zip(
                old_p.shape,
                new_p.shape
            ):
                min_shape.append(
                    min(d1, d2)
                )

            if len(min_shape) == 2:

                old_p = old_p[
                    :min_shape[0],
                    :min_shape[1]
                ]

                new_p = new_p[
                    :min_shape[0],
                    :min_shape[1]
                ]

            elif len(min_shape) == 1:

                old_p = old_p[
                    :min_shape[0]
                ]

                new_p = new_p[
                    :min_shape[0]
                ]

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
# ACTIVATION HOOK
# ============================================================

def build_activation_hook(name):

    def hook(module, inp, out):

        if isinstance(out, tuple):
            out = out[0]

        activation_storage[name] = (
            out.detach()
            .abs()
            .mean()
            .item()
        )

    return hook

# ============================================================
# REGISTER ACTIVATION HOOKS
# ============================================================

def register_activation_hooks(model):

    hooks = []

    for name, module in model.named_modules():

        if "ffn" in name.lower():

            h = module.register_forward_hook(
                build_activation_hook(name)
            )

            hooks.append(h)

    return hooks

# ============================================================
# BUILD GRAD HOOK
# ============================================================

def build_grad_hook(name):

    def hook(grad):

        if name not in locked_masks:
            return grad

        mask = locked_masks[name].to(
            grad.device
        )

        return grad * mask

    return hook

# ============================================================
# REGISTER HOOK LOCK
# ============================================================

def register_hook_lock(model):

    hooks = []

    for name, param in model.named_parameters():

        # ====================================================
        # ONLY FFN
        # ====================================================

        if "ffn" not in name.lower():
            continue

        # ====================================================
        # ONLY TRAINABLE
        # ====================================================

        if not isinstance(param, nn.Parameter):
            continue

        if not param.requires_grad:
            continue

        # ====================================================
        # INIT AGE
        # ====================================================

        if name not in neuron_age_map:

            neuron_age_map[name] = 0

        # ====================================================
        # INIT MASK
        # ====================================================

        if name not in locked_masks:

            locked_masks[name] = torch.ones_like(param)

        # ====================================================
        # REGISTER
        # ====================================================

        try:

            h = param.register_hook(
                build_grad_hook(name)
            )

            hooks.append(h)

        except Exception as e:

            print(
                f"[HOOK SKIP] "
                f"{name} -> {e}"
            )

    return hooks

# ============================================================
# UPDATE AGE LOCK
# ============================================================

def update_age_lock(model):

    for name, param in model.named_parameters():

        if "ffn" not in name.lower():
            continue

        if not isinstance(param, nn.Parameter):
            continue

        if not param.requires_grad:
            continue

        if param.ndim < 2:
            continue

        current_dim = param.shape[0]

        old_dim = neuron_age_map.get(
            name,
            0
        )

        mask = torch.ones_like(param)

        # ====================================================
        # LOCK OLD REGION
        # ====================================================

        if old_dim > 0:

            lock_size = min(
                old_dim,
                current_dim
            )

            mask[:lock_size] = 0.0

        locked_masks[name] = mask

        neuron_age_map[name] = current_dim

# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(
    use_hook_lock=False
):

    title = (
        "HookLock"
        if use_hook_lock
        else "NoLock"
    )

    print("\n===================================================")
    print(f"🚨 RUNNING {title}")
    print("===================================================\n")

    # ========================================================
    # MODEL
    # ========================================================

    config = ModelConfig(
        vocab_size=16000,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=8,
        initial_ffn_dim=256,
        max_seq_len=2048,
    )

    model = GrowableLLM(config).to(DEVICE)

    # ========================================================
    # HOOKS
    # ========================================================

    lock_hooks = []

    activation_hooks = register_activation_hooks(model)

    if use_hook_lock:

        lock_hooks = register_hook_lock(model)

        update_age_lock(model)

    # ========================================================
    # METRICS
    # ========================================================

    all_task_losses = []

    forgetting_scores = []

    drift_history = []

    localization_history = []

    sparsity_history = []

    params_history = []

    results = []

    previous_state = save_model_state(model)

    # ========================================================
    # TRAIN
    # ========================================================

    for stage in range(NUM_DOMAINS):

        print("\n===================================================")
        print(f"🧠 STAGE {stage+1}")
        print(f"📚 DOMAIN = {DOMAIN_NAMES[stage]}")
        print("===================================================\n")

        # ====================================================
        # EXPAND
        # ====================================================

        if stage > 0:

            print(
                f"📈 动态扩容 FFN: +{EXPAND_DIM}"
            )

            model.expand_model(
                extra_dim=EXPAND_DIM
            )

            # ================================================
            # REMOVE OLD HOOKS
            # ================================================

            for h in lock_hooks:
                h.remove()

            lock_hooks = []

            # ================================================
            # REBUILD HOOK
            # ================================================

            if use_hook_lock:

                lock_hooks = register_hook_lock(model)

                update_age_lock(model)

        # ====================================================
        # OPTIMIZER
        # ====================================================

        optimizer = torch.optim.AdamW(
            filter(
                lambda p: p.requires_grad,
                model.parameters()
            ),
            lr=LR
        )

        model.train()

        # ====================================================
        # TRAIN LOOP
        # ====================================================

        for step in tqdm(
            range(TRAIN_STEPS),
            desc=f"{title}-{DOMAIN_NAMES[stage]}"
        ):

            x = generate_domain_batch(
                config.vocab_size,
                stage
            )

            optimizer.zero_grad()

            _, loss = model(
                x,
                labels=x
            )

            loss.backward()

            optimizer.step()

            if step % 25 == 0:

                print(
                    f"Step={step:03d} "
                    f"Loss={loss.item():.6f}"
                )

        # ====================================================
        # EVAL
        # ====================================================

        current_losses = []

        print("\n📊 Evaluating")

        for eval_id in range(NUM_DOMAINS):

            eval_loss = evaluate_domain(
                model,
                config.vocab_size,
                eval_id
            )

            current_losses.append(
                eval_loss
            )

            print(
                f"{DOMAIN_NAMES[eval_id]} "
                f"Loss={eval_loss:.6f}"
            )

        all_task_losses.append(
            current_losses
        )

        # ====================================================
        # FORGETTING
        # ====================================================

        current_forgetting = []

        for old_task in range(stage):

            old_best = all_task_losses[
                old_task
            ][old_task]

            current_perf = current_losses[
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

        # ====================================================
        # DRIFT
        # ====================================================

        current_state = save_model_state(model)

        drift = compute_old_memory_drift(
            previous_state,
            current_state
        )

        drift_history.append(drift)

        previous_state = current_state

        # ====================================================
        # ACTIVATION
        # ====================================================

        activation_storage.clear()

        with torch.no_grad():

            x = generate_domain_batch(
                config.vocab_size,
                stage
            )

            _ = model(
                x,
                labels=x
            )

        activation_values = np.array(
            list(
                activation_storage.values()
            )
        )

        if len(activation_values) > 0:

            localization = np.std(
                activation_values
            )

            sparsity = np.mean(
                activation_values
                < np.mean(activation_values)
            )

        else:

            localization = 0.0

            sparsity = 0.0

        localization_history.append(
            localization
        )

        sparsity_history.append(
            sparsity
        )

        # ====================================================
        # PARAMS
        # ====================================================

        params = count_parameters(model)

        params_history.append(params)

        # ====================================================
        # SAVE
        # ====================================================

        result = {

            "stage": stage,

            "domain": DOMAIN_NAMES[stage],

            "params": params,

            "avg_forgetting": avg_forgetting,

            "old_memory_drift": drift,

            "knowledge_localization": localization,

            "routing_sparsity": sparsity,

            "avg_loss": np.mean(
                current_losses
            ),
        }

        results.append(result)

        print("\n---------------------------------------------------")

        print(
            f"Forgetting = "
            f"{avg_forgetting:.6f}"
        )

        print(
            f"Drift = "
            f"{drift:.6f}"
        )

        print(
            f"Localization = "
            f"{localization:.6f}"
        )

        print(
            f"Sparsity = "
            f"{sparsity:.6f}"
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    for h in lock_hooks:
        h.remove()

    for h in activation_hooks:
        h.remove()

    return {

        "results": results,

        "losses": all_task_losses,

        "forgetting": forgetting_scores,

        "drift": drift_history,

        "localization": localization_history,

        "sparsity": sparsity_history,
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # NO LOCK
    # ========================================================

    nolock = run_experiment(
        use_hook_lock=False
    )

    # ========================================================
    # RESET
    # ========================================================

    locked_masks.clear()

    neuron_age_map.clear()

    # ========================================================
    # HOOK LOCK
    # ========================================================

    hooklock = run_experiment(
        use_hook_lock=True
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    nolock_df = pd.DataFrame(
        nolock["results"]
    )

    hooklock_df = pd.DataFrame(
        hooklock["results"]
    )

    nolock_df.to_csv(
        os.path.join(
            SAVE_DIR,
            "nolock.csv"
        ),
        index=False
    )

    hooklock_df.to_csv(
        os.path.join(
            SAVE_DIR,
            "hooklock.csv"
        ),
        index=False
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    comparison = pd.DataFrame({

        "Method": [
            "NoLock",
            "HookLock",
        ],

        "Avg Forgetting": [
            np.mean(
                nolock["forgetting"]
            ),
            np.mean(
                hooklock["forgetting"]
            ),
        ],

        "Avg Drift": [
            np.mean(
                nolock["drift"]
            ),
            np.mean(
                hooklock["drift"]
            ),
        ],

        "Localization": [
            np.mean(
                nolock["localization"]
            ),
            np.mean(
                hooklock["localization"]
            ),
        ],

        "Routing Sparsity": [
            np.mean(
                nolock["sparsity"]
            ),
            np.mean(
                hooklock["sparsity"]
            ),
        ],
    })

    comparison.to_csv(
        os.path.join(
            SAVE_DIR,
            "comparison.csv"
        ),
        index=False
    )

    print("\n===================================================")
    print("📊 FINAL COMPARISON")
    print("===================================================\n")

    print(comparison)

    # ========================================================
    # FORGETTING PLOT
    # ========================================================

    plt.figure(figsize=(10,6))

    plt.plot(
        nolock["forgetting"],
        marker='o',
        linewidth=3,
        label="NoLock"
    )

    plt.plot(
        hooklock["forgetting"],
        marker='o',
        linewidth=3,
        label="HookLock"
    )

    plt.title(
        "Catastrophic Forgetting",
        fontsize=18
    )

    plt.xlabel("Stage")

    plt.ylabel("Forgetting")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "forgetting.png"
        ),
        dpi=300,
        bbox_inches='tight'
    )

    # ========================================================
    # DRIFT
    # ========================================================

    plt.figure(figsize=(10,6))

    plt.plot(
        nolock["drift"],
        marker='o',
        linewidth=3,
        label="NoLock"
    )

    plt.plot(
        hooklock["drift"],
        marker='o',
        linewidth=3,
        label="HookLock"
    )

    plt.title(
        "Old Memory Drift",
        fontsize=18
    )

    plt.xlabel("Stage")

    plt.ylabel("Drift")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "drift.png"
        ),
        dpi=300,
        bbox_inches='tight'
    )

    # ========================================================
    # HEATMAP
    # ========================================================

    plt.figure(figsize=(12,8))

    sns.heatmap(
        np.array(nolock["losses"]),
        annot=True,
        cmap="magma",
        xticklabels=DOMAIN_NAMES,
        yticklabels=[
            f"After_{i+1}"
            for i in range(NUM_DOMAINS)
        ]
    )

    plt.title("NoLock Matrix")

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "nolock_matrix.png"
        ),
        dpi=300,
        bbox_inches='tight'
    )

    plt.figure(figsize=(12,8))

    sns.heatmap(
        np.array(hooklock["losses"]),
        annot=True,
        cmap="viridis",
        xticklabels=DOMAIN_NAMES,
        yticklabels=[
            f"After_{i+1}"
            for i in range(NUM_DOMAINS)
        ]
    )

    plt.title("HookLock Matrix")

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "hooklock_matrix.png"
        ),
        dpi=300,
        bbox_inches='tight'
    )

    print("\n===================================================")
    print("✅ ALL EXPERIMENTS FINISHED")
    print("===================================================\n")