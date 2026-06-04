import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from model import GrowableLLM, ModelConfig

# =====================================================
# [配置区]
# =====================================================
CHAT_DATA_PATH = "daily_chat_clean.jsonl"
BASE_WEIGHT_PATH = "growable_qwen_base.pth" # 纯净基座
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4       
MAX_LENGTH = 768     
EPOCHS = 3           
LEARNING_RATE = 1e-4 

# =====================================================
# 1. 人情层数据集构建
# =====================================================
class ChatDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print("📖 正在加载并解析高情商对话语料...")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): 
                    self.data.append(json.loads(line.strip()))
        print(f"✅ 成功加载 {len(self.data)} 条语料。")

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = (
            f"<|im_start|>system\n你是一个高情商、乐于助人的AI助手。<|im_end|>\n"
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

# =====================================================
# 🌟 修复点：将整理函数封装为全局类，完美兼容 Windows 多进程
# =====================================================
class ChatCollate:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch):
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        # 动态 Padding
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=self.pad_token_id)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return inputs_padded, labels_padded

# =====================================================
# 2. 第一阶段生长引擎
# =====================================================
def main():
    print(f"🚀 [小乐协议-第一拓展] 高情商脑区生长启动！设备: {DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = ModelConfig(
        vocab_size=151936, hidden_dim=1024, num_layers=24, 
        num_heads=16, num_kv_heads=16, initial_ffn_dim=2816
    )
    model = GrowableLLM(config).to(DEVICE)
    
    print(f"💉 注入通用世界观基座: {BASE_WEIGHT_PATH}...")
    model.load_state_dict(torch.load(BASE_WEIGHT_PATH, map_location=DEVICE, weights_only=True), strict=False)
    
    model.expand_model(extra_dim=256)
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    
    dataset = ChatDataset(CHAT_DATA_PATH, tokenizer, MAX_LENGTH)
    
    # 🌟 使用重构后的 ChatCollate
    collate_fn = ChatCollate(tokenizer.pad_token_id)

    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,        
        pin_memory=True,      
        prefetch_factor=2     
    )
    
    total_steps = len(loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda')
    
    print("\n" + "="*50)
    print("🔒 基座已绝对锁死！所有梯度将仅注入新增的 256 维突触！")
    print(f"📊 开始进行 {EPOCHS} 轮高情商语料特训...")
    print("="*50 + "\n")
    
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Chat Epoch {epoch+1}/{EPOCHS}")
        
        for step, (input_ids, labels) in enumerate(progress_bar):
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                _, loss = model(input_ids, labels=labels)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            
            if step % 10 == 0:
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
                
        avg_loss = total_loss / len(loader)
        print(f"🎉 Epoch {epoch+1} 结束！平均 Loss: {avg_loss:.4f} | 显存: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        
    SAVE_PATH = "growable_chat_expert_epoch3.pth"
    torch.save(model.state_dict(), SAVE_PATH)
    print("\n🏆 高情商中枢生长完毕！权重已固化至:", SAVE_PATH)
    print("下一步：你可以带着这个模型，去跑医学脑区的 Defrag 融合拓展了！")

if __name__ == "__main__":
    main()