"""
Qwen1.5-0.5B Phase 2: Medical Expert Training + Defrag (HF backbone)

Uses GrowableQwen2ForCausalLM instead of hand-written GrowableLLM.
Optimized for 8GB VRAM: batch_size=1, gradient_accumulation=32.
"""

import os
import sys
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

# Windows GBK 兼容
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "buffer"):
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

# 确保能导入根目录的 models_hf.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models_hf import GrowableQwen2ForCausalLM

# =====================================================
# [配置区]
# =====================================================
MED_DATA_PATH = "domain_2_medical_cleaned.jsonl"
CHAT_DATA_PATH = "daily_chat_clean_cleaned.jsonl"  # 用于 defrag 回放
BASE_WEIGHT_PATH = "growable_qwen_chat_expert_epoch3.pt"  # Phase 1 产出
MODEL_ID = "Qwen/Qwen1.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 8GB VRAM 适配
BATCH_SIZE = 1
MAX_LENGTH = 1024
EPOCHS = 3
EXTRA_DIM = 128
LEARNING_RATE = 1e-4
ACCUMULATION_STEPS = 32

# =====================================================
# 1. 数据集构建
# =====================================================
class DomainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, system_prompt, limit=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.data = []

        print(f"Loading data: {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))

        if limit:
            random.shuffle(self.data)
            self.data = self.data[:limit]

        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{item.get('instruction', '')}\n{item.get('input', '')}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        response = f"{item.get('output', '')}<|im_end|>"

        p_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        r_ids = self.tokenizer.encode(response, add_special_tokens=False)

        input_ids = (p_ids + r_ids)[:self.max_length]
        labels = [-100] * len(p_ids) + r_ids
        labels = labels[:self.max_length]

        if all(l == -100 for l in labels):
            labels[-1] = input_ids[-1]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DynamicCollate:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=self.pad_token_id)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return inputs_padded, labels_padded


# =====================================================
# 2. Phase 2: Medical Expert Training
# =====================================================
def main():
    print(f"[Phase 2] Medical Expert Training | Device: {DEVICE}")

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1. 加载基座并恢复 Phase 1 权重
    model = GrowableQwen2ForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    print(f"Base model loaded! Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    print(f"Loading Phase 1 weights: {BASE_WEIGHT_PATH}...")
    ckpt = torch.load(BASE_WEIGHT_PATH, map_location=DEVICE, weights_only=True)

    # 自动探测 Phase 1 扩容的维度，并对齐骨架
    # Qwen1.5-0.5B 初始 FFN dim = 2816
    initial_ffn_dim = 2816
    sample_key = "model.layers.0.mlp.gate_proj.weight"
    if sample_key in ckpt:
        loaded_ffn_dim = ckpt[sample_key].shape[0]
        need_expand = loaded_ffn_dim - initial_ffn_dim
        if need_expand > 0:
            model.expand_model(extra_dim=need_expand)
            print(f"Auto-expanded +{need_expand} dim to match Phase 1 checkpoint")

    model.load_state_dict(ckpt, strict=True)
    print(f"Phase 1 weights restored! Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    collate_fn = DynamicCollate(tokenizer.pad_token_id)

    # =====================================================
    # Stage A: Defrag (不对称解锁碎片整理)
    # =====================================================
    print(f"\n{'='*50}")
    print("Stage A: Defrag - aligning new units with attention distributions...")
    print(f"{'='*50}")

    # 取 32 条聊天数据作为回放锚点
    replay_dataset = DomainDataset(
        CHAT_DATA_PATH,
        tokenizer,
        MAX_LENGTH,
        system_prompt="You are a helpful AI assistant.",
        limit=32
    )
    replay_loader = DataLoader(replay_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 手动执行 defrag（使用 HF backbone 的 defrag API）
    # 先准备融合数据：取一个 batch 的 input_ids
    fusion_input_ids, _ = next(iter(replay_loader))
    fusion_input_ids = fusion_input_ids.to(DEVICE)

    # 配置优化器用于 defrag
    defrag_opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-6
    )

    model.defrag(defrag_opt, fusion_data=fusion_input_ids, target_replay_size=32)
    print("Defrag complete! Top-layer attention aligned.")

    # =====================================================
    # Stage B: 物理扩容 + 医学突触生长
    # =====================================================
    print(f"\n{'='*50}")
    print(f"Stage B: Expanding +{EXTRA_DIM} dim for Medical domain...")
    print(f"{'='*50}")

    # expand_model 内部会 global_lock=True，锁死之前所有参数
    model.expand_model(extra_dim=EXTRA_DIM)
    print(f"Expanded! Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    med_optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)

    med_prompt = (
        "You are an expert in Traditional Chinese Medicine (TCM). "
        "You can accurately diagnose based on symptoms and provide rigorous herbal prescriptions."
    )
    med_dataset = DomainDataset(MED_DATA_PATH, tokenizer, MAX_LENGTH, system_prompt=med_prompt)

    med_loader = DataLoader(
        med_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    total_steps = len(med_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        med_optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps
    )

    print(f"\nHookLock active! Only new {EXTRA_DIM}-dim medical synapses receive gradients.")
    print(f"Training: {EPOCHS} epochs, {len(med_dataset)} samples, grad_accum={ACCUMULATION_STEPS}\n")

    model.train()
    torch.set_float32_matmul_precision('high')

    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(med_loader, desc=f"Medical Epoch {epoch+1}/{EPOCHS}")
        med_optimizer.zero_grad()

        for step, (input_ids, labels) in enumerate(progress_bar):
            input_ids, labels = input_ids.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                _, loss = model(input_ids, labels=labels)

            (loss / ACCUMULATION_STEPS).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                med_optimizer.step()
                scheduler.step()
                med_optimizer.zero_grad()

            total_loss += loss.item()
            if step % 10 == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

        # 清空剩余梯度
        if (step + 1) % ACCUMULATION_STEPS != 0:
            med_optimizer.step()
            med_optimizer.zero_grad()

        avg_loss = total_loss / len(med_loader)
        vram_mb = torch.cuda.memory_allocated() / (1024**2)
        print(f"Epoch {epoch+1} done! Avg Loss: {avg_loss:.4f} | VRAM: {vram_mb:.0f} MB")

    # 最终 defrag：融合医学突触与已有注意力
    print("\nFinal defrag: aligning medical synapses...")
    fusion_input_ids, _ = next(iter(replay_loader))
    fusion_input_ids = fusion_input_ids.to(DEVICE)
    final_defrag_opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-6
    )
    model.defrag(final_defrag_opt, fusion_data=fusion_input_ids, target_replay_size=32)

    SAVE_PATH = "growable_qwen_med_expert_epoch3.pt"
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nMedical expert weights saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
