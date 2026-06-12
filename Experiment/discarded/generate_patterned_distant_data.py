import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models import GrowableLLM, ModelConfig

def generate_concentrated_pattern_data(vocab_size, domain_id, batch_size=4, seq_len=32):
    """
    【有规律且高度集中的数据】
    - 公共规律：基础的序列递增/循环结构。
    - 集中分布：把之前步长 150 改成 15，让不同领域的词表分布紧密挨在一起，
      逼迫顶层在极其拥挤的高维空间里处理细微的特征冲突。
    """
    torch.manual_seed(domain_id * 9999)
    data = torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    # 构造底层公共规律：比如 0, 1, 2, 3, 4 的周期循环
    base_pattern = torch.arange(seq_len) % 5 
    
    for b in range(batch_size):
        # 🌟 修改点 1：将巨大的偏移量缩小 (比如步长设为 15)，让领域间距变得非常集中
        domain_shift = (domain_id * 15) % (vocab_size - 50)
        
        # 最终数据 = 共享语法规律 * 局部跨度 + 专属领地偏移 + 微小局部随机扰动
        noise = torch.randint(0, 5, (seq_len,))
        data[b] = base_pattern * 5 + domain_shift + noise
        
    return data

def run_snapshot_interference_test():
    print("🚀 启动 [真·纯融合无回放] 集中规律数据 快照切面测试...")
    
    config = ModelConfig(
        vocab_size=10000, hidden_dim=128, num_layers=2, 
        num_heads=4, num_kv_heads=4, initial_ffn_dim=256
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GrowableLLM(config).to(device)
    
    NUM_DOMAINS = 50
    EXTRA_DIM_PER_DOMAIN = 16
    TRAIN_STEPS = 5
    
    print("📚 正在构建高度集中的规律数据集...")
    train_datasets = [generate_concentrated_pattern_data(config.vocab_size, i).to(device) for i in range(NUM_DOMAINS)]
    
    snapshot_results = {}

    for current_stage in tqdm(range(NUM_DOMAINS), desc="年轮生长进度"):
        
        # 1. 物理扩容
        if current_stage > 0:
            model.expand_model(extra_dim=EXTRA_DIM_PER_DOMAIN)

        # 2. 训练当前新领域
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.0)
        
        current_data = train_datasets[current_stage]
        
        model.train()
        for _ in range(TRAIN_STEPS):
            optimizer.zero_grad()
            _, loss = model(current_data, labels=current_data)
            loss.backward()
            optimizer.step()
            
        # =====================================================
        # ⚠️ 【无回放】强行融合，真融合修复！
        # =====================================================
        # 🌟 修改点 2：必须【先】解开顶层的物理封印
        unlock_start_layer = max(0, len(model.blocks) - 6)
        for i in range(unlock_start_layer, len(model.blocks)):
            for param in model.blocks[i].parameters():
                param.requires_grad = True
        model.norm.weight.requires_grad = True
        
        # 🌟 然后【再】抓取 requires_grad=True 的参数建立新优化器。
        # 此时，顶层矩阵才真正进入了 optimizer 的更新名单！
        defrag_params = filter(lambda p: p.requires_grad, model.parameters())
        real_defrag_opt = torch.optim.AdamW(defrag_params, lr=1e-3)
        
        real_defrag_opt.zero_grad()
        _, defrag_loss = model(current_data, labels=current_data)
        if defrag_loss is not None and not (torch.isnan(defrag_loss) or torch.isinf(defrag_loss)):
            defrag_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            real_defrag_opt.step()
        # =====================================================

        # =====================================================
        # 🔬 每满 5 个领域，给整个 AI 拍一张快照
        # =====================================================
        num_mounted = current_stage + 1
        if num_mounted % 5 == 0:
            model.eval()
            current_snapshot_losses = []
            
            with torch.no_grad():
                for domain_id in range(num_mounted):
                    eval_data = train_datasets[domain_id]
                    _, eval_loss = model(eval_data, labels=eval_data)
                    current_snapshot_losses.append(eval_loss.item())
                    
            snapshot_results[num_mounted] = current_snapshot_losses

    # =====================================================
    # 🌟 修改点 3：重命名输出的表和图
    # =====================================================
    print("\n📊 正在生成量化表格...")
    df_data = {}
    for stage, losses in snapshot_results.items():
        row_data = {}
        for i, loss in enumerate(losses):
            row_data[f"Domain_{i+1}"] = loss
        df_data[f"Snapshot_{stage}"] = row_data

    df = pd.DataFrame.from_dict(df_data, orient='index')
    domain_cols = [f"Domain_{i}" for i in range(1, NUM_DOMAINS + 1) if f"Domain_{i}" in df.columns]
    df = df[domain_cols]
    df.index.name = "AI_Snapshot_Stage"
    
    # 导出为新的 CSV 名字
    df.to_csv("concentrated_fusion_matrix.csv")
    print("✅ 量化表格已保存至 concentrated_fusion_matrix.csv")

    # ---------------- 绘图 ----------------
    plt.figure(figsize=(14, 8))
    
    colors = sns.color_palette("turbo", len(snapshot_results))
    
    for i, (stage, losses) in enumerate(snapshot_results.items()):
        x_axis = list(range(1, stage + 1)) 
        
        plt.plot(
            x_axis, losses, 
            marker='o', linestyle='-', linewidth=2, markersize=6,
            color=colors[i], label=f"loss{stage} (AI with {stage} domains)"
        )

    plt.title("Concentrated Pattern Snapshot Profile - True Cross-Interference Visualization", fontsize=16)
    plt.xlabel("Evaluated Domain ID (1 to 50)", fontsize=14)
    plt.ylabel("Validation Loss", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.legend(fontsize=11, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    # 保存为新的图片名字
    plt.savefig("concentrated_fusion_test.png", dpi=300)
    print("✅ 集中规律数据的快照切面图已保存至 concentrated_fusion_test.png")

if __name__ == "__main__":
    run_snapshot_interference_test()