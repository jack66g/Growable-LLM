# ============================================================
# DEFRAG NECESSITY BENCHMARK (Final Corrected Version)
# V1: No Defrag | V2: Full Defrag | V3: Attention Defrag
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

# 从你不变的 model.py 中导入
from models import GrowableLLM, ModelConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "outputs/defrag_benchmark"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_DOMAINS = 10
DOMAIN_NAMES = [
    "Medical", "Law", "Finance", "Code", "Math", 
    "ChineseWiki", "EnglishWiki", "Reasoning", "LongContext", "Instruction"
]

PHASE1_STEPS = 100   # 扩容后仅训练新维度的步数
PHASE2_STEPS = 30    # 顶层 Defrag 的融合步数
BATCH_SIZE = 8
SEQ_LEN = 64
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4     
EXPAND_DIM = 64
TOP_K_LAYERS = 2     

# 全局变量：用于物理隔离
locked_masks = {}
neuron_age_map = {}

# ============================================================
# DATA & EVAL
# ============================================================
def generate_domain_batch(vocab_size, domain_id, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
    x = torch.zeros((batch_size, seq_len), dtype=torch.long)
    domain_seed = domain_id * 9999 + random.randint(0, 9999)
    torch.manual_seed(domain_seed)
    base_pattern = torch.arange(seq_len)
    domain_shift = domain_id * 1300
    for b in range(batch_size):
        noise = torch.randint(0, 50, (seq_len,))
        tokens = (base_pattern * (domain_id + 17) + domain_shift + noise) % vocab_size
        x[b] = tokens
    return x.to(DEVICE)

@torch.no_grad()
def evaluate_domain(model, vocab_size, domain_id):
    model.eval()
    losses = []
    for _ in range(5):
        x = generate_domain_batch(vocab_size, domain_id)
        # 调用原生的 model，不传递任何额外参数
        _, loss = model(x, labels=x)
        losses.append(loss.item())
    return np.mean(losses)

# ============================================================
# HOOK LOCK MECHANISM (The Core Physical Isolation)
# ============================================================
def build_grad_hook(name):
    def hook(grad):
        if name in locked_masks:
            mask = locked_masks[name].to(grad.device)
            return grad * mask
        return grad
    return hook

def register_hook_lock(model):
    hooks = []
    for name, param in model.named_parameters():
        if "ffn" not in name.lower() or not param.requires_grad:
            continue
        if name not in neuron_age_map:
            neuron_age_map[name] = 0
        if name not in locked_masks:
            locked_masks[name] = torch.ones_like(param)
        try:
            hooks.append(param.register_hook(build_grad_hook(name)))
        except:
            pass
    return hooks

def update_age_lock(model):
    for name, param in model.named_parameters():
        if "ffn" not in name.lower() or not param.requires_grad or param.ndim < 2:
            continue
        current_dim = param.shape[0]
        old_dim = neuron_age_map.get(name, 0)
        mask = torch.ones_like(param)
        if old_dim > 0:
            lock_size = min(old_dim, current_dim)
            mask[:lock_size] = 0.0
        locked_masks[name] = mask
        neuron_age_map[name] = current_dim

# ============================================================
# PHASE 2: DEFRAG PROTOCOLS (The Variable)
# ============================================================
def set_defrag_requires_grad(model, num_layers, variant):
    """根据 V1/V2/V3 变体，设置顶层融合阶段的梯度锁"""
    if variant == "V1_No_Defrag":
        for p in model.parameters(): p.requires_grad = False
        return

    # 先冻结全模型
    for p in model.parameters(): p.requires_grad = False
    top_layers_idx = [num_layers - i - 1 for i in range(TOP_K_LAYERS)]
    
    for name, param in model.named_parameters():
        # 始终解锁最终的 Norm 层
        if "norm" in name.lower() and "layers" not in name:
            param.requires_grad = True
            
        for idx in top_layers_idx:
            layer_prefix = f"layers.{idx}." 
            if layer_prefix in name:
                if variant == "V2_Full_Defrag":
                    # V2 完全体：解锁顶层的 Attention + FFN + Norm
                    param.requires_grad = True
                elif variant == "V3_Attention_Defrag":
                    # V3 对照组：只解锁顶层的 Attention 和层内 Norm，锁死 FFN
                    if "attn" in name.lower() or "attention" in name.lower() or "norm" in name.lower():
                        param.requires_grad = True

# ============================================================
# RUN VARIANT EXPERIMENT
# ============================================================
def run_variant(variant_name):
    print(f"\n{'='*65}\n🚀 RUNNING BENCHMARK: {variant_name}\n{'='*65}")
    
    config = ModelConfig(vocab_size=16000, hidden_dim=256, num_layers=6, num_heads=8, num_kv_heads=8, initial_ffn_dim=256, max_seq_len=2048)
    model = GrowableLLM(config).to(DEVICE)
    
    locked_masks.clear()
    neuron_age_map.clear()
    lock_hooks = []
    
    metrics = {
        "stage": [],
        "domain": [],
        "relative_forgetting_pct": [],
        "fwt_gain": [],
        "cross_interference": []
    }
    loss_matrix = []
    
    # ----------------------------------------------------
    # Step -1: Record Pure Random Baseline for FWT
    # ----------------------------------------------------
    print("🎲 Evaluating Pure Random Baseline...")
    random_baseline_losses = [evaluate_domain(model, config.vocab_size, d) for d in range(NUM_DOMAINS)]

    for stage in range(NUM_DOMAINS):
        print(f"\n--- 🧠 STAGE {stage+1}/{NUM_DOMAINS} | DOMAIN: {DOMAIN_NAMES[stage]} ---")
        
        # Expand Architecture
        if stage > 0:
            model.expand_model(extra_dim=EXPAND_DIM)
            for h in lock_hooks: h.remove()
        
        lock_hooks = register_hook_lock(model)
        update_age_lock(model)

        # ----------------------------------------------------
        # PHASE 1: Orthogonal Growth (Only train new FFN dims)
        # ----------------------------------------------------
        for p in model.parameters(): p.requires_grad = False
        for name, param in model.named_parameters():
            if "ffn" in name.lower(): param.requires_grad = True 
                
        optimizer_p1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1)
        model.train()
        for step in tqdm(range(PHASE1_STEPS), desc=f"{variant_name} Phase 1 (Grow)"):
            x = generate_domain_batch(config.vocab_size, stage)
            optimizer_p1.zero_grad()
            _, loss = model(x, labels=x)
            loss.backward()
            optimizer_p1.step()

        # ----------------------------------------------------
        # PHASE 2: Defragmentation (Fusion based on variant)
        # ----------------------------------------------------
        if variant_name != "V1_No_Defrag":
            set_defrag_requires_grad(model, config.num_layers, variant_name)
            optimizer_p2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE2)
            model.train()
            for step in tqdm(range(PHASE2_STEPS), desc=f"{variant_name} Phase 2 (Defrag)"):
                x = generate_domain_batch(config.vocab_size, stage)
                optimizer_p2.zero_grad()
                _, loss = model(x, labels=x)
                loss.backward()
                optimizer_p2.step()

        # ----------------------------------------------------
        # EVALUATION & REVISED METRICS
        # ----------------------------------------------------
        current_losses = [evaluate_domain(model, config.vocab_size, d) for d in range(NUM_DOMAINS)]
        loss_matrix.append(current_losses)
        
        # 修正1：Relative Forgetting % -> ((Current - Best) / Best * 100)
        forget_pct = 0.0
        if stage > 0:
            past_forget_pcts = []
            for t in range(stage):
                best_t = loss_matrix[t][t]
                current_t = current_losses[t]
                past_forget_pcts.append((current_t - best_t) / best_t)
            forget_pct = np.mean(past_forget_pcts) * 100.0
            
        # 修正2：FWT Gain -> (Random Baseline Loss - Current Future Loss)
        fwt_gain = 0.0
        if stage < NUM_DOMAINS - 1:
            future_gains = []
            for future_t in range(stage + 1, NUM_DOMAINS):
                gain = random_baseline_losses[future_t] - current_losses[future_t]
                future_gains.append(gain)
            fwt_gain = np.mean(future_gains)
            
        # 修正3：Cross-Interference -> (Current Task Loss - Loss Before Current Training Phase)
        cross_interference = 0.0
        if stage > 0:
            interferences = []
            for t in range(stage):
                # 训练当前领域后，对历史领域导致的绝对污染增量
                interference = current_losses[t] - loss_matrix[stage-1][t]
                interferences.append(interference)
            cross_interference = np.mean(interferences)

        metrics["stage"].append(stage)
        metrics["domain"].append(DOMAIN_NAMES[stage])
        metrics["relative_forgetting_pct"].append(forget_pct)
        metrics["fwt_gain"].append(fwt_gain)
        metrics["cross_interference"].append(cross_interference)
        
        print(f"📊 Forgetting: {forget_pct:+.2f}% | FWT Gain: {fwt_gain:+.4f} | Cross-Interference: {cross_interference:+.4f}")

    for h in lock_hooks: h.remove()
    return pd.DataFrame(metrics), np.array(loss_matrix)

# ============================================================
# MAIN EXECUTION & VISUALIZATION
# ============================================================
if __name__ == "__main__":
    variants = ["V1_No_Defrag", "V2_Full_Defrag", "V3_Attention_Defrag"]
    results_dfs, loss_matrices = {}, {}
    
    for v in variants:
        df, mat = run_variant(v)
        results_dfs[v] = df
        loss_matrices[v] = mat
        df.to_csv(os.path.join(SAVE_DIR, f"{v}_metrics.csv"), index=False)
        
    summary_data = []
    for v in variants:
        df = results_dfs[v]
        summary_data.append({
            "Method": v,
            "Avg Forgetting % (↓)": df["relative_forgetting_pct"].mean(),
            "Avg FWT Gain (↑)": df["fwt_gain"].mean(),
            "Avg Cross-Interference (↓)": df["cross_interference"].mean()
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(SAVE_DIR, "Final_Defrag_Benchmark.csv"), index=False)
    
    print("\n" + "="*65)
    print("🏆 FINAL BENCHMARK RESULTS")
    print("="*65)
    print(summary_df.to_string(index=False))

    # ==================== 绘图 ====================
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # 1. Forgetting % 曲线
    plt.figure(figsize=(10, 6))
    for v in variants:
        plt.plot(results_dfs[v]["stage"], results_dfs[v]["relative_forgetting_pct"], marker='o', linewidth=2.5, label=v)
    plt.title("Relative Forgetting % over 10 Domains (Lower is better)", fontsize=14, fontweight='bold')
    plt.xlabel("Domain Stage")
    plt.ylabel("Forget (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "defrag_forgetting_pct.png"), dpi=300)

    # 2. 泛化增益曲线 (FWT)
    plt.figure(figsize=(10, 6))
    for v in variants:
        plt.plot(results_dfs[v]["stage"], results_dfs[v]["fwt_gain"], marker='s', linewidth=2.5, label=v)
    plt.title("Forward Transfer (FWT) Gain vs Random Baseline (Higher is better)", fontsize=14, fontweight='bold')
    plt.xlabel("Domain Stage")
    plt.ylabel("Loss Reduction on Future Tasks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "defrag_fwt_gain.png"), dpi=300)

    # 3. $10 \times 10$ 交叉干扰矩阵热力图
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for i, v in enumerate(variants):
        sns.heatmap(loss_matrices[v], ax=axes[i], cmap="viridis", annot=False,
                    xticklabels=DOMAIN_NAMES, yticklabels=[f"After_{d}" for d in DOMAIN_NAMES])
        axes[i].set_title(f"{v}\n10x10 Interference Matrix", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "defrag_10x10_matrices.png"), dpi=300)

    print(f"\n✅ All Benchmark completed! Results and plots saved to '{SAVE_DIR}'.")