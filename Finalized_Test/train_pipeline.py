import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm

# Windows GBK 兼容：确保 stdout 支持 UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "buffer"):
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

# 确保能导入根目录的 models.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入我们手写的完美架构
from models import GrowableLLM, ModelConfig

# ==========================================
# 从统一配置文件读取
# ==========================================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    full_config = json.load(f)

model_cfg = full_config["model"]
train_cfg = full_config["training"]
hw_cfg = full_config["hardware"]["profiles"][full_config["hardware"]["profile"]]
tk_cfg = full_config["tokenizer"]
paths = full_config["paths"]

# 将配置中的相对路径解析为相对于项目根目录的绝对路径
for key in ("base_weights", "master_weights", "magicoder_data", "evolcode_data"):
    paths[key] = os.path.join(PROJECT_ROOT, paths[key])

DEVICE = "cuda"
BATCH_SIZE = hw_cfg["batch_size"]
SEQ_LEN = hw_cfg["seq_len"]
LR = train_cfg["learning_rate"]
EXPAND_DIM = train_cfg["expand_dim"]
EPOCHS_PER_PHASE = train_cfg["epochs_per_phase"]
NUM_WORKERS = hw_cfg["num_workers"]
ACCUMULATION_STEPS = hw_cfg["gradient_accumulation_steps"]
MAX_SAMPLES = train_cfg["max_samples_per_phase"][full_config["hardware"]["profile"]]

# ==========================================
# 数据处理与 Tokenizer
# ==========================================
os.environ["HF_ENDPOINT"] = tk_cfg["hf_endpoint"]
tokenizer = AutoTokenizer.from_pretrained(tk_cfg["model_id"], local_files_only=True)
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
    return input_ids, labels  # CPU 张量，由训练循环移到 GPU

# ==========================================
# 训练引擎
# ==========================================
def run_training_phase(model, phase_name, dataset, extra_dim):
    print(f"\n{'='*60}")
    print(f"🔥 STARTING PHASE: {phase_name}")
    print(f"{'='*60}")

    # 1. 执行正交扩容 (自动分配新矩阵并锁死旧知识)
    model.expand_model(extra_dim=extra_dim)

    # 2. 准备 DataLoader（num_workers=0 避免 CUDA 多进程传输瓶颈）
    # 限制样本数以控制训练时长
    total_rows = len(dataset)
    n_samples = min(MAX_SAMPLES, total_rows)
    if n_samples < total_rows:
        dataset = dataset.shuffle(seed=42).select(range(n_samples))
        print(f"  Subsampled to {n_samples} rows (from {total_rows} total) for this hardware profile")
    else:
        print(f"  Using full dataset: {total_rows} rows")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # 3. 配置优化器 (此时 require_grad=True 的只有新扩容的矩阵部分)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    
    # 4. 开始暴力训练 (使用 Bfloat16 混合精度，RTX 6000 的杀手锏)
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=False) # BF16 不需要 scaler，直接 autocast
    
    for epoch in range(EPOCHS_PER_PHASE):
        pbar = tqdm(dataloader, desc=f"{phase_name} Epoch {epoch+1}")
        total_loss = 0
        optimizer.zero_grad()

        for step, (input_ids, labels) in enumerate(pbar):
            # CPU->GPU 传输（不在 collate_fn 中做，以兼容 num_workers)
            input_ids, labels = input_ids.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            # 开启 AMP Bfloat16 大幅加速计算
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                logits, loss = model(input_ids, labels=labels)

            # 梯度累积：除 ACCUMULATION_STEPS 实现小 batch 等效大 batch
            (loss / ACCUMULATION_STEPS).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            if step % 10 == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # 清空剩余梯度
        if (step + 1) % ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()
                
    print(f"✅ {phase_name} 扩容训练完成! 平均 Loss: {total_loss/len(dataloader):.4f}")
    
    # 5. 触发 V3 Attention Defrag 协议 (使用一小批数据对齐 Attention)
    # 取一个 Batch 的数据进行特征融合
    fusion_input_ids, _ = next(iter(dataloader))
    model.defrag(optimizer, fusion_data=fusion_input_ids.to(DEVICE))
    
    print(f"🔒 {phase_name} 阶段的知识已完美固化！")

# ==========================================
# 终极总指挥部 (Main)
# ==========================================
if __name__ == "__main__":
    # 开启 CUDNN 基准测试，让 GPU 自动寻找最快卷积算法
    torch.backends.cudnn.benchmark = True

    # 1. 从配置文件读取模型架构
    config = ModelConfig(**model_cfg)

    model = GrowableLLM(config).to(DEVICE)

    # 2. 载入原始基座权重
    print(f"⏳ 载入基座权重: {paths['base_weights']}")
    model.load_state_dict(torch.load(paths["base_weights"]))
    print("✅ 基座载入完毕！")

    # 3. 载入本地数据集
    ds_logic = load_from_disk(paths["magicoder_data"])
    ds_code = load_from_disk(paths["evolcode_data"])

    # ================= 激进训练开始 =================

    # Phase 1: 纯逻辑推演注入
    run_training_phase(model, phase_name="[Stage 1: Logic Reasoning]", dataset=ds_logic, extra_dim=EXPAND_DIM)

    # Phase 2: 代码语法与生成注入
    run_training_phase(model, phase_name="[Stage 2: Code Generation]", dataset=ds_code, extra_dim=EXPAND_DIM)

    # ==============================================

    # 4. 保存最终的终极生命体
    final_path = paths["master_weights"]
    torch.save(model.state_dict(), final_path)
    print(f"\n🎉 连环正交扩展全部完成！终极权重已保存至: {final_path}")