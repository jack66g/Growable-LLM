import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models import GrowableLLM, ModelConfig

def generate_isolated_random_data(vocab_size, domain_id, batch_size=4, seq_len=32):
    """绝对隔离的无规律随机数据，迫使模型记忆，杜绝周期性巧合"""
    torch.manual_seed(domain_id * 9999) 
    data = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    return data

def run_snapshot_interference_test():
    print("🚀 启动 [纯融合无回放] 年轮快照切面交叉感染测试...")
    
    config = ModelConfig(
        vocab_size=2000, hidden_dim=128, num_layers=2, 
        num_heads=4, num_kv_heads=4, initial_ffn_dim=256
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GrowableLLM(config).to(device)
    
    NUM_DOMAINS = 50
    EXTRA_DIM_PER_DOMAIN = 16
    TRAIN_STEPS = 5
    
    print("📚 正在构建纯随机的无规律数据集...")
    train_datasets = [generate_isolated_random_data(config.vocab_size, i).to(device) for i in range(NUM_DOMAINS)]
    
    # 用一个字典来保存每 5 轮的“AI 整体快照”在各个领域上的表现
    # 格式: { 5: [loss_1, loss_2... loss_5], 10: [loss_1... loss_10] }
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
        # ⚠️ 【无回放】强行融合，只看当前数据
        # =====================================================
        # 【🎯 核心 Bug 修复区】：
        # 提前解冻顶层参数，确保新建的优化器能把顶层参数抓进名单里
        unlock_start_layer = max(0, len(model.blocks) - 6)
        for i in range(unlock_start_layer, len(model.blocks)):
            for param in model.blocks[i].parameters():
                param.requires_grad = True
        model.norm.weight.requires_grad = True
        
        # 新建一个真正包含顶层权重的“融合优化器”
        fusion_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

        # 把包含顶层参数的新优化器传进去，触发真正的融合更新！
        model.defrag(fusion_optimizer, current_data)
        # =====================================================

        # =====================================================
        # 🔬 严格按照你的逻辑：每满 5 个领域，给整个 AI 拍一张快照
        # =====================================================
        num_mounted = current_stage + 1
        if num_mounted % 5 == 0:
            model.eval()
            current_snapshot_losses = []
            
            with torch.no_grad():
                # 拿这个长了 num_mounted 个矩阵的整体 AI，去逐个测它已有的领域
                for domain_id in range(num_mounted):
                    eval_data = train_datasets[domain_id]
                    _, eval_loss = model(eval_data, labels=eval_data)
                    current_snapshot_losses.append(eval_loss.item())
                    
            # 存下这条名为 lossX 的线的所有点
            snapshot_results[num_mounted] = current_snapshot_losses

    # =====================================================
    # 🌟 新增：将字典数据量化为 Pandas DataFrame 并导出 CSV
    # =====================================================
    print("\n📊 正在生成量化表格...")
    df_data = {}
    for stage, losses in snapshot_results.items():
        row_data = {}
        for i, loss in enumerate(losses):
            row_data[f"Domain_{i+1}"] = loss
        df_data[f"Snapshot_{stage}"] = row_data

    # 转换为 DataFrame，以 Snapshot 为行，Domain 为列
    df = pd.DataFrame.from_dict(df_data, orient='index')
    
    # 确保列按照 Domain_1, Domain_2... 的顺序排列
    domain_cols = [f"Domain_{i}" for i in range(1, NUM_DOMAINS + 1) if f"Domain_{i}" in df.columns]
    df = df[domain_cols]
    df.index.name = "AI_Snapshot_Stage"
    
    # 导出为 CSV
    df.to_csv("snapshot_interference_matrix.csv")
    print("✅ 量化表格已保存至 snapshot_interference_matrix.csv")
    # =====================================================

    # ---------------- 绘图：完美的快照切面图 ----------------
    plt.figure(figsize=(14, 8))
    
    colors = sns.color_palette("turbo", len(snapshot_results))
    
    # 遍历每个阶段的快照，画出你说的 loss5, loss10 等等
    for i, (stage, losses) in enumerate(snapshot_results.items()):
        # X 轴是领域的编号：从 1 到当前 stage
        x_axis = list(range(1, stage + 1)) 
        
        plt.plot(
            x_axis, losses, 
            marker='o', linestyle='-', linewidth=2, markersize=6,
            color=colors[i], label=f"loss{stage} (AI with {stage} domains)"
        )

    plt.title("Model Snapshot Profile (No Replay Fusion) - Cross-Interference Visualization", fontsize=16)
    plt.xlabel("Evaluated Domain ID (1 to 50)", fontsize=14)
    plt.ylabel("Validation Loss", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 调整图例位置
    plt.legend(fontsize=11, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("snapshot_interference_test.png", dpi=300)
    print("✅ 真正的快照切面图已保存至 snapshot_interference_test.png")

if __name__ == "__main__":
    run_snapshot_interference_test()