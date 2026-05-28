import os
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from model import GrowableLLM, ModelConfig

# =====================================================
# [配置区]
# =====================================================
MED_DATA_PATH = "domain_2_medical.jsonl"
CHAT_DATA_PATH = "daily_chat_clean.jsonl" # 🌟 用于提取回放锚点
BASE_WEIGHT_PATH = "growable_chat_expert_epoch3.pth" # 挂载第一阶段的高情商大盘
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4
MAX_LENGTH = 1024
EPOCHS = 3
EXTRA_DIM = 128      # 🌟 为赛博老中医开辟 128 维专属脑区
LEARNING_RATE = 1e-4

# =====================================================
# 1. 数据集构建与动态 Collate
# =====================================================
class DomainDataset(Dataset):
    # 🌟 新增 system_prompt 参数，作为唤醒特定脑区的密钥
    def __init__(self, data_path, tokenizer, max_length, system_prompt, limit=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.data = []
        
        print(f"📖 正在加载语料: {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): 
                    self.data.append(json.loads(line.strip()))
        
        if limit:
            random.shuffle(self.data)
            self.data = self.data[:limit]
            
        print(f"✅ 成功加载 {len(self.data)} 条数据。")

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 🌟 动态注入专属的人格提示词！
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
# 2. 第二阶段：融合与生长引擎
# =====================================================
def main():
    print(f"🚀 [小乐协议-第二拓展] 赛博老中医特训启动！设备: {DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1. 恢复高情商大盘
    config = ModelConfig(
        vocab_size=151936, hidden_dim=1024, num_layers=24, 
        num_heads=16, num_kv_heads=16, initial_ffn_dim=2816
    )
    model = GrowableLLM(config).to(DEVICE)
    
    print(f"💉 正在读取高情商大盘: {BASE_WEIGHT_PATH}...")
    ckpt = torch.load(BASE_WEIGHT_PATH, map_location=DEVICE, weights_only=True)
    
    # 自动探测第一阶段生长的 256 维，并对齐骨架
    target_ffn_dim = ckpt["blocks.0.ffn.gate_proj.weight"].shape[0]
    need_expand = target_ffn_dim - config.initial_ffn_dim
    if need_expand > 0:
        model.expand_model(extra_dim=need_expand)
        
    model.load_state_dict(ckpt, strict=True)
    
    collate_fn = DynamicCollate(tokenizer.pad_token_id)

    # =====================================================
    # 🌟 阶段 A：可视化 Defrag (不对称解锁碎片整理)
    # =====================================================
    print("\n" + "="*50)
    print("🌀 启动 [不对称解锁 Defrag] 回放融合阶段...")
    print("="*50)
    
    # 🌟 提取 32 条聊天数据作为回放锚点，锚定 256 维情商层！
    replay_dataset = DomainDataset(
        CHAT_DATA_PATH, 
        tokenizer, 
        MAX_LENGTH, 
        system_prompt="你是一个高情商、乐于助人的AI助手。",
        limit=32
    )
    replay_loader = DataLoader(replay_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # 全局死锁，然后精准解锁最后 6 层和 Norm
    for param in model.parameters(): param.requires_grad = False
    for i in range(max(0, len(model.blocks) - 6), len(model.blocks)):
        for param in model.blocks[i].parameters(): param.requires_grad = True
    model.norm.weight.requires_grad = True
    
    # 极低学习率进行融合
    defrag_opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)
    
    model.train()
    for step, (input_ids, labels) in enumerate(tqdm(replay_loader, desc="Defrag 回放进度")):
        input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
        defrag_opt.zero_grad()
        with torch.amp.autocast('cuda'):
            _, loss = model(input_ids, labels=labels)
        loss.backward()
        defrag_opt.step()
        
        # 🌟 让你亲眼看到融合 Loss！
        if step % 2 == 0:
            print(f"  [Defrag 监控] Replay Loss: {loss.item():.4f}")
            
    print("✨ 顶层网络与情商中枢融合完毕！地基已彻底打牢！")

    # =====================================================
    # 🌟 阶段 B：物理扩容与中医突触生长
    # =====================================================
    print("\n" + "="*50)
    print(f"🧬 启动物理扩容：挂载 {EXTRA_DIM} 维医学突触...")
    print("="*50)
    
    # expand_model 内部会强制调用 global_lock=True，将刚才融合好的底盘彻底锁死防弹！
    model.expand_model(extra_dim=EXTRA_DIM)
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    med_optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    
    # 🌟 真正的赛博华佗提示词！这将会成为唤醒 128 维矩阵的终极密钥！
    med_prompt = "你是一位精通传统中医（TCM）的专家。你能根据患者的症状，准确辨证论治，并给出严谨的中药或方剂建议。"
    med_dataset = DomainDataset(MED_DATA_PATH, tokenizer, MAX_LENGTH, system_prompt=med_prompt)
    
    med_loader = DataLoader(
        med_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True, prefetch_factor=2
    )
    
    total_steps = len(med_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(med_optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n📊 绝对隔离环境就绪！开始 {EPOCHS} 轮中医语料特训...")
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(med_loader, desc=f"Medical Epoch {epoch+1}/{EPOCHS}")
        
        for step, (input_ids, labels) in enumerate(progress_bar):
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            med_optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                _, loss = model(input_ids, labels=labels)
                
            scaler.scale(loss).backward()
            scaler.unscale_(med_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(med_optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            
            if step % 10 == 0:
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
                
        avg_loss = total_loss / len(med_loader)
        print(f"🎉 Epoch {epoch+1} 结束！平均 Loss: {avg_loss:.4f} | 显存: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        
    SAVE_PATH = "growable_med_expert_epoch3.pth"
    torch.save(model.state_dict(), SAVE_PATH)
    print("\n🏆 赛博老中医脑区固化完毕！完美隔离大盘已保存至:", SAVE_PATH)

if __name__ == "__main__":
    main()