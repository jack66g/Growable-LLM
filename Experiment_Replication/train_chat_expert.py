"""
Qwen1.5-0.5B Phase 1: Chat Expert Training (HF backbone)

Uses GrowableQwen2ForCausalLM instead of hand-written GrowableLLM.
Optimized for 8GB VRAM: batch_size=1, gradient_accumulation=32.
"""

import os
import sys
import json
import torch
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
CHAT_DATA_PATH = "daily_chat_clean_cleaned.jsonl"
MODEL_ID = "Qwen/Qwen1.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 8GB VRAM 适配：bs=1 + 大梯度累积
BATCH_SIZE = 1
MAX_LENGTH = 768
EPOCHS = 3
LEARNING_RATE = 1e-4
EXPAND_DIM = 256
ACCUMULATION_STEPS = 32  # 等效 batch_size = 1 * 32 = 32

# =====================================================
# 1. 数据集构建
# =====================================================
class ChatDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        print("Loading chat corpus...")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = (
            f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
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


class ChatCollate:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=self.pad_token_id)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return inputs_padded, labels_padded


# =====================================================
# 2. Phase 1: Chat Expert Training
# =====================================================
def main():
    print(f"[Phase 1] Chat Expert Training | Device: {DEVICE}")

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 从 HF 加载预训练模型（自动切换 DynamicSwiGLUMLP）
    model = GrowableQwen2ForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    print(f"Base model loaded! Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 正交扩容 256 维
    model.expand_model(extra_dim=EXPAND_DIM)
    print(f"Expanded +{EXPAND_DIM} dim! Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)

    dataset = ChatDataset(CHAT_DATA_PATH, tokenizer, MAX_LENGTH)
    collate_fn = ChatCollate(tokenizer.pad_token_id)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows 兼容
        pin_memory=True,
    )

    total_steps = len(loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps
    )

    print(f"\n{'='*50}")
    print(f"HookLock active! Only new {EXPAND_DIM}-dim synapses receive gradients.")
    print(f"Training: {EPOCHS} epochs, {len(dataset)} samples, grad_accum={ACCUMULATION_STEPS}")
    print(f"{'='*50}\n")

    model.train()
    torch.set_float32_matmul_precision('high')

    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Chat Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad()

        for step, (input_ids, labels) in enumerate(progress_bar):
            input_ids, labels = input_ids.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                _, loss = model(input_ids, labels=labels)

            (loss / ACCUMULATION_STEPS).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            if step % 10 == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

        # 清空剩余梯度
        if (step + 1) % ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(loader)
        vram_mb = torch.cuda.memory_allocated() / (1024**2)
        print(f"Epoch {epoch+1} done! Avg Loss: {avg_loss:.4f} | VRAM: {vram_mb:.0f} MB")

    SAVE_PATH = "growable_qwen_chat_expert_epoch3.pt"
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nChat expert weights saved to: {SAVE_PATH}")
    print("Next: run train_med_expert.py for Phase 2 (Medical domain + defrag)")


if __name__ == "__main__":
    main()
