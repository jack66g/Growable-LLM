import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm

# 导入我们手写的完美架构
from models import GrowableLLM, ModelConfig

# ==========================================
# RTX 6000 Pro "大火力" 配置参数
# ==========================================
DEVICE = "cuda"
BATCH_SIZE = 32          # 96GB 显存，直接拉满！(如果显存还有余，甚至可以 128)
SEQ_LEN = 4096           # 足够覆盖绝大多数逻辑推演和代码
LR = 2e-4                # 扩容层学习率可以稍微大一点
EXPAND_DIM = 256         # 每次正交扩容 256 维 (克制且高效)
EPOCHS_PER_PHASE = 1     # 验证想法先跑 1 个 Epoch
NUM_WORKERS = 8          # 多线程拉取数据，不让 GPU 等待 CPU

# ==========================================
# 数据处理与 Tokenizer
# ==========================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def collate_fn(batch):
    # 将 Instruction 和 Output 组装成 ChatML 格式
    texts = []
    for item in batch:
        instruction = item.get('instruction', '')
        # Evol-Code 的标签是 'output'，Magicoder 是 'response'，我们做个兼容
        response = item.get('output', item.get('response', '')) 
        
        text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
        texts.append(text)
        
    encoded = tokenizer(
        texts, 
        truncation=True, 
        max_length=SEQ_LEN, 
        padding="max_length", 
        return_tensors="pt"
    )
    
    input_ids = encoded["input_ids"]
    labels = input_ids.clone()
    
    # 遮蔽 (Mask) 掉 Padding 部分，不参与 Loss 计算
    labels[encoded["attention_mask"] == 0] = -100
    return input_ids.to(DEVICE), labels.to(DEVICE)

# ==========================================
# 训练引擎
# ==========================================
def run_training_phase(model, phase_name, dataset, extra_dim):
    print(f"\n{'='*60}")
    print(f"🔥 STARTING PHASE: {phase_name}")
    print(f"{'='*60}")
    
    # 1. 执行正交扩容 (自动分配新矩阵并锁死旧知识)
    model.expand_model(extra_dim=extra_dim)
    
    # 2. 准备大火力 DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True # 加速显存转移
    )
    
    # 3. 配置优化器 (此时 require_grad=True 的只有新扩容的矩阵部分)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    
    # 4. 开始暴力训练 (使用 Bfloat16 混合精度，RTX 6000 的杀手锏)
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=False) # BF16 通常不需要 scaler，直接 autocast
    
    for epoch in range(EPOCHS_PER_PHASE):
        pbar = tqdm(dataloader, desc=f"{phase_name} Epoch {epoch+1}")
        total_loss = 0
        
        for step, (input_ids, labels) in enumerate(pbar):
            optimizer.zero_grad()
            
            # 开启 AMP Bfloat16 大幅加速计算
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                logits, loss = model(input_ids, labels=labels)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            if step % 10 == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
    print(f"✅ {phase_name} 扩容训练完成! 平均 Loss: {total_loss/len(dataloader):.4f}")
    
    # 5. 触发 V3 Attention Defrag 协议 (使用一小批数据对齐 Attention)
    # 取一个 Batch 的数据进行特征融合
    fusion_input_ids, _ = next(iter(dataloader))
    model.defrag(optimizer, fusion_data=fusion_input_ids)
    
    print(f"🔒 {phase_name} 阶段的知识已完美固化！")

# ==========================================
# 终极总指挥部 (Main)
# ==========================================
if __name__ == "__main__":
    # 开启 CUDNN 基准测试，让 RTX 6000 自动寻找最快卷积算法
    torch.backends.cudnn.benchmark = True 
    
    # 1. 实例化 SmolLM2-360M 配置
    config = ModelConfig(
        vocab_size=49152, hidden_dim=960, num_layers=32, 
        num_heads=15, num_kv_heads=5, initial_ffn_dim=2560, rope_theta=100000
    )
    
    model = GrowableLLM(config).to(DEVICE)
    
    # 2. 载入原始基座权重
    print("⏳ 载入 SmolLM2 基座权重...")
    model.load_state_dict(torch.load("smollm2_360m_growable.pt"))
    print("✅ 基座载入完毕！")
    
    # 3. 载入本地数据集
    ds_logic = load_from_disk("growable_llm_data/magicoder_110k")
    ds_code = load_from_disk("growable_llm_data/evol_code_80k")
    
    # ================= 激进训练开始 =================
    
    # Phase 1: 纯逻辑推演注入 (+256维)
    run_training_phase(model, phase_name="[Stage 1: Logic Reasoning]", dataset=ds_logic, extra_dim=EXPAND_DIM)
    
    # Phase 2: 代码语法与生成注入 (+256维)
    run_training_phase(model, phase_name="[Stage 2: Code Generation]", dataset=ds_code, extra_dim=EXPAND_DIM)
    
    # ==============================================
    
    # 4. 保存最终的终极生命体
    final_path = "GrowableLLM_360M_LogicCode_Master.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n🎉 连环正交扩展全部完成！终极权重已保存至: {final_path}")