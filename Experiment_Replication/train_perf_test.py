"""
性能测试脚本：运行 1 分钟后自动终止
用于测试 VRAM、CPU、内存占用
"""

import os
import sys
import time
import torch
import psutil
import threading
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "buffer"):
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models_hf import GrowableQwen2ForCausalLM

# =====================================================
# 配置
# =====================================================
CHAT_DATA_PATH = "daily_chat_clean_cleaned.jsonl"
MODEL_ID = "Qwen/Qwen1.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1
MAX_LENGTH = 768
EPOCHS = 1
LEARNING_RATE = 1e-4
EXPAND_DIM = 256
ACCUMULATION_STEPS = 32
RUNTIME_LIMIT = 60  # 1 分钟后自动终止

# =====================================================
# 数据集（复用 train_chat_expert 的预处理逻辑）
# =====================================================
class ChatDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        import json
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        print("Loading and tokenizing chat corpus...")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    prompt = (
                        f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
                        f"<|im_start|>user\n{item.get('instruction', '')}\n{item.get('input', '')}<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )
                    response = f"{item.get('output', '')}<|im_end|>"

                    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    r_ids = tokenizer.encode(response, add_special_tokens=False)

                    input_ids = (p_ids + r_ids)[:max_length]
                    labels = [-100] * len(p_ids) + r_ids
                    labels = labels[:max_length]

                    if all(l == -100 for l in labels):
                        labels[-1] = input_ids[-1]

                    self.data.append((input_ids, labels))

        print(f"Tokenized {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, labels = self.data[idx]
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
# 性能监控
# =====================================================
class PerformanceMonitor:
    def __init__(self):
        self.running = True
        self.stats = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        process = psutil.Process()
        while self.running:
            try:
                vram = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
                vram_reserved = torch.cuda.memory_reserved() / (1024**2) if torch.cuda.is_available() else 0
                mem = process.memory_info().rss / (1024**2)
                cpu = process.cpu_percent()

                self.stats.append({
                    'time': time.time(),
                    'vram_mb': vram,
                    'vram_reserved_mb': vram_reserved,
                    'mem_mb': mem,
                    'cpu_percent': cpu
                })
            except:
                pass
            time.sleep(1)

    def stop(self):
        self.running = False
        self.thread.join()

    def report(self):
        if not self.stats:
            return "No stats recorded"

        vram_max = max(s['vram_mb'] for s in self.stats)
        vram_avg = sum(s['vram_mb'] for s in self.stats) / len(self.stats)
        mem_max = max(s['mem_mb'] for s in self.stats)
        mem_avg = sum(s['mem_mb'] for s in self.stats) / len(self.stats)
        cpu_max = max(s['cpu_percent'] for s in self.stats)
        cpu_avg = sum(s['cpu_percent'] for s in self.stats) / len(self.stats)

        return f"""
========================================
性能报告 (采集 {len(self.stats)} 秒)
========================================
VRAM 峰值: {vram_max:.1f} MB
VRAM 平均: {vram_avg:.1f} MB
内存 峰值: {mem_max:.1f} MB
内存 平均: {mem_avg:.1f} MB
CPU 峰值: {cpu_max:.1f}%
CPU 平均: {cpu_avg:.1f}%
========================================
"""


# =====================================================
# 主程序
# =====================================================
def main():
    print(f"[Perf Test] Qwen Training | Device: {DEVICE}")
    print(f"[Perf Test] Will auto-kill after {RUNTIME_LIMIT} seconds")
    print("=" * 50)

    start_time = time.time()

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = GrowableQwen2ForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    print(f"Model loaded! Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

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
        num_workers=0,
        pin_memory=True,
    )

    total_steps = len(loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps
    )

    print(f"\nTraining config: {EPOCHS} epochs, {len(dataset)} samples, grad_accum={ACCUMULATION_STEPS}")
    print(f"Estimated time per step: ~2-3s")
    print(f"Estimated steps in {RUNTIME_LIMIT}s: ~{RUNTIME_LIMIT//3}\n")

    # 启动性能监控
    monitor = PerformanceMonitor()

    model.train()
    torch.set_float32_matmul_precision('high')

    progress_bar = tqdm(loader, desc="Training")
    optimizer.zero_grad()

    step_count = 0
    try:
        for step, (input_ids, labels) in enumerate(progress_bar):
            # 检查是否超时
            elapsed = time.time() - start_time
            if elapsed >= RUNTIME_LIMIT:
                print(f"\n⏰ Time limit ({RUNTIME_LIMIT}s) reached! Stopping...")
                break

            input_ids, labels = input_ids.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                _, loss = model(input_ids, labels=labels)

            (loss / ACCUMULATION_STEPS).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step_count = step + 1
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'elapsed': f"{elapsed:.0f}s"
            })

        # 清理最后的梯度
        if step_count % ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")

    # 停止监控并输出报告
    monitor.stop()
    print(monitor.report())

    elapsed = time.time() - start_time
    print(f"Total steps: {step_count}, Time: {elapsed:.1f}s, Speed: {step_count/elapsed:.2f} steps/s")


if __name__ == "__main__":
    main()
