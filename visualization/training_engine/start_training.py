import os
import sys
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models import GrowableLLM, ModelConfig

class Color:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# =========================================================
# 物理张量外科手术 (动态扩容)
# =========================================================
def perform_tensor_surgery(model, expand_dim):
    print(f"[{Color.GREEN}Surgery{Color.RESET}] 开始执行物理张量拼接手术 (为每一层 FFN +{expand_dim} 维)...")
    
    for i, block in enumerate(model.blocks):
        old_gate = block.ffn.gate_proj
        old_up = block.ffn.up_proj
        old_down = block.ffn.down_proj
        
        in_features = old_gate.in_features
        old_ffn_dim = old_gate.out_features
        new_ffn_dim = old_ffn_dim + expand_dim
        
        new_gate = nn.Linear(in_features, new_ffn_dim, bias=False).to(old_gate.weight.dtype)
        new_up = nn.Linear(in_features, new_ffn_dim, bias=False).to(old_up.weight.dtype)
        new_down = nn.Linear(new_ffn_dim, in_features, bias=False).to(old_down.weight.dtype)
        
        with torch.no_grad():
            new_gate.weight[:old_ffn_dim, :] = old_gate.weight
            nn.init.normal_(new_gate.weight[old_ffn_dim:, :], mean=0.0, std=0.02)
            
            new_up.weight[:old_ffn_dim, :] = old_up.weight
            nn.init.normal_(new_up.weight[old_ffn_dim:, :], mean=0.0, std=0.02)
            
            new_down.weight[:, :old_ffn_dim] = old_down.weight
            nn.init.zeros_(new_down.weight[:, old_ffn_dim:])
        
        block.ffn.gate_proj = new_gate
        block.ffn.up_proj = new_up
        block.ffn.down_proj = new_down

# =========================================================
# 核心：真实的碎片整理协议 (Defrag Protocol)
# =========================================================
def execute_defrag(model, strategy, current_data_tensors, memory_pool_path, device, tokenizer):
    print(f"\n{Color.CYAN}={'='*50}={Color.RESET}")
    print(f"{Color.BOLD}🚀 触发碎片整理 (Defrag Protocol): {strategy}{Color.RESET}")
    
    if strategy == "NONE":
        print(f"{Color.YELLOW}策略 [NONE]: 常规微调，跳过碎片整理，直接结束。{Color.RESET}")
        return

    for param in model.parameters():
        param.requires_grad = False
        
    total_layers = len(model.blocks)
    start_unlock = max(0, total_layers - 6)
    print(f"[*] 正在物理锁死底层 {start_unlock} 层，仅解锁顶层 6 层路由通道 (Attention & Norm)...")
    
    for i in range(start_unlock, total_layers):
        for param in model.blocks[i].attn.parameters():
            param.requires_grad = True
        for param in model.blocks[i].attn_norm.parameters():
            param.requires_grad = True

    fusion_tensors = []
    
    if strategy == "CURRENT_ONLY":
        print(f"[*] 策略 [CURRENT_ONLY]: 从当前知识抽取 2 条数据微调顶层...")
        if current_data_tensors:
            fusion_tensors.extend(random.sample(current_data_tensors, min(2, len(current_data_tensors))))
        
    elif strategy == "REPLAY_2":
        print(f"[*] 策略 [REPLAY_2]: 启动历史盲盒回放机制！")
        current_2 = []
        if current_data_tensors:
            current_2 = random.sample(current_data_tensors, min(2, len(current_data_tensors)))
            fusion_tensors.extend(current_2)
        
        if os.path.exists(memory_pool_path):
            with open(memory_pool_path, 'r', encoding='utf-8') as f:
                history_texts = [json.loads(line)['text'] for line in f]
            
            if history_texts:
                hist_2_texts = random.sample(history_texts, min(2, len(history_texts)))
                for text in hist_2_texts:
                    tokens = tokenizer.encode(text, return_tensors="pt")[0]
                    fusion_tensors.append(tokens)
                print(f"[*] 成功从历史盲盒中抓取 {len(hist_2_texts)} 条旧知识锚点！")
        else:
            print(f"[*] 历史盲盒为空，首次拓展，仅使用当前数据。")
            
        if current_2:
            with open(memory_pool_path, 'a', encoding='utf-8') as f:
                for tokens in current_2:
                    text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
            print(f"[*] 已将当前领域锚点投递至历史盲盒。")

    if fusion_tensors:
        defrag_optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)
        print(f"[{Color.GREEN}INFO{Color.RESET}] 开始执行真实的单步时空融合...")
        
        defrag_loss_total = 0
        valid_batches = 0
        
        for seq_tensor in fusion_tensors:
            if len(seq_tensor) <= 1:
                continue
                
            seq_tensor = seq_tensor.unsqueeze(0).to(device)
            logits = model(seq_tensor)
            
            if isinstance(logits, tuple):
                logits = logits[0]
            elif hasattr(logits, 'logits'):
                logits = logits.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = seq_tensor[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss.backward()
            defrag_loss_total += loss.item()
            valid_batches += 1
            
        if valid_batches > 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= valid_batches
            defrag_optimizer.step()
            defrag_optimizer.zero_grad()
            print(f"{Color.GREEN}✅ Defrag 特征对齐完毕！融合 Loss: {defrag_loss_total/valid_batches:.4f}{Color.RESET}")

def main():
    payload_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train_payload.json")
    if not os.path.exists(payload_path):
        print(f"{Color.RED}❌ 找不到 train_payload.json，请先通过主控台生成配置！{Color.RESET}")
        sys.exit(1)
        
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
        
    train_cfg = payload['training']
    base_cfg = payload['base_model']
    
    print(f"\n{Color.BOLD}{Color.CYAN}╔══════════════════════════════════════════════════╗{Color.RESET}")
    print(f"{Color.BOLD}{Color.CYAN}║         🔥 GrowableLLM 真实炼丹引擎启动 🔥       ║{Color.RESET}")
    print(f"{Color.BOLD}{Color.CYAN}╚══════════════════════════════════════════════════╝{Color.RESET}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{Color.GREEN}System{Color.RESET}] GPU 计算节点就绪: {device}")

    # =========================================================
    # 物理沙盒重建与权重挂载
    # =========================================================
    print(f"[{Color.GREEN}System{Color.RESET}] 正在根据 Payload 重塑物理骨架...")
    config = ModelConfig(
        vocab_size=base_cfg['vocab_size'],
        hidden_dim=base_cfg['hidden_dim'],
        num_layers=base_cfg['num_layers'],
        num_heads=base_cfg['num_heads'],
        num_kv_heads=base_cfg['num_kv_heads'],
        initial_ffn_dim=base_cfg['initial_ffn_dim']
    )
    
    model = GrowableLLM(config)
    model.load_state_dict(torch.load(base_cfg['model_path'], map_location="cpu", weights_only=True))
    
    expand_dim = train_cfg['expand_dim']
    perform_tensor_surgery(model, expand_dim)
    model.to(device)

    tokenizer_path = base_cfg.get('tokenizer_path')
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        print(f"{Color.RED}❌ 找不到本地 Tokenizer ({tokenizer_path})！{Color.RESET}")
        sys.exit(1)
        
    print(f"[*] 正在挂载专属本地 Tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # =========================================================
    # 🌟 智能万能数据加载与严格清洗
    # =========================================================
    print(f"[{Color.GREEN}System{Color.RESET}] 正在解析并 Tokenize 训练数据...")
    valid_data_tensors = []
    discard_count = 0
    empty_count = 0
    max_ctx = train_cfg['max_context']
    
    with open(train_cfg['data_path'], 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            
            text = ""
            try:
                data_dict = json.loads(line)
                # 智能识别不同数据集格式
                if 'text' in data_dict:
                    text = data_dict['text']
                elif 'content' in data_dict:
                    text = data_dict['content']
                elif 'messages' in data_dict:  # OpenAI 格式
                    msgs = data_dict['messages']
                    if isinstance(msgs, list):
                        text = "\n".join([m.get('content', '') for m in msgs if isinstance(m, dict)])
                elif 'instruction' in data_dict: # Alpaca 格式
                    text = data_dict.get('instruction', '') + "\n" + data_dict.get('output', '')
                else:
                    # 暴力兜底：拼接所有字符串
                    text = " ".join([str(v) for v in data_dict.values() if isinstance(v, str)])
            except:
                # 不是标准 JSON，直接当纯文本处理
                text = line.strip()

            if not text.strip():
                empty_count += 1
                continue

            tokens = tokenizer.encode(text, return_tensors="pt")[0]
            seq_len = len(tokens)
            
            # 严格拦截：长度小于等于 1 (无法算 Loss) 或者 超长 (OOM风险)
            if seq_len <= 1:
                empty_count += 1
                continue
            if seq_len > max_ctx:
                discard_count += 1
                continue
                
            valid_data_tensors.append(tokens)

    print(f"[*] 扫描完毕！有效数据: {Color.GREEN}{len(valid_data_tensors)}{Color.RESET} 条 | 抛弃超长: {Color.RED}{discard_count}{Color.RESET} | 丢弃空数据: {Color.YELLOW}{empty_count}{Color.RESET}")
    
    if not valid_data_tensors:
        print(f"{Color.RED}❌ 有效训练数据为 0！所有数据均被拦截（可能全是空数据或超长数据），强制终止。{Color.RESET}")
        sys.exit(1)

    # =========================================================
    # 真实训练主循环 (Phase 1)
    # =========================================================
    print(f"\n{Color.CYAN}={'='*50}={Color.RESET}")
    print(f"{Color.BOLD}🚀 阶段一：扩容脑区真实前向传播训练...{Color.RESET}")
    
    model.train()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_cfg['learning_rate'])
    
    batch_size = train_cfg['batch_size']
    start_time = time.time()
    
    for step, tokens in enumerate(valid_data_tensors, 1):
        seq_len = len(tokens)
        
        tokens = tokens.unsqueeze(0).to(device)
        logits = model(tokens)
        
        if isinstance(logits, tuple):
            logits = logits[0]
        elif hasattr(logits, 'logits'):
            logits = logits.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tokens[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        (loss / batch_size).backward()
        
        if step % batch_size == 0 or step == len(valid_data_tensors):
            optimizer.step()
            optimizer.zero_grad()

        elapsed = time.time() - start_time
        speed = step / elapsed if elapsed > 0 else 0
        sys.stdout.write(
            f"\r{Color.YELLOW}[炼丹中]{Color.RESET} "
            f"Step: {step}/{len(valid_data_tensors)} | "
            f"Len: {Color.CYAN}{seq_len:^4}{Color.RESET} | "
            f"Loss: {Color.GREEN}{loss.item():.4f}{Color.RESET} | "
            f"Speed: {speed:.1f} it/s"
        )
        sys.stdout.flush()
        
        if step % train_cfg['save_steps'] == 0:
            print(f"\n💾 [Checkpoint] 达到 {step} 步，正在保存...")
            torch.save(model.state_dict(), f"checkpoint_step_{step}.pt")

    print(f"\n{Color.GREEN}✅ 阶段一训练完成！耗时: {time.time() - start_time:.1f}s{Color.RESET}")

    # =========================================================
    # 执行碎片整理 (Phase 2)
    # =========================================================
    memory_pool_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history_memory_pool.jsonl")
    execute_defrag(model, train_cfg['defrag_strategy'], valid_data_tensors, memory_pool_path, device, tokenizer)

    # =========================================================
    # 最终落盘
    # =========================================================
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_weights")
    os.makedirs(output_dir, exist_ok=True)
    
    final_name = f"Growable_Dim{expand_dim}_{train_cfg['defrag_strategy']}.pt"
    save_path = os.path.join(output_dir, final_name)
    
    # 1. 保存模型权重 (.pt)
    torch.save(model.state_dict(), save_path)
    
    # 2. 🌟 生成并保存伴生的 _config.json
    new_ffn_dim = base_cfg['initial_ffn_dim'] + expand_dim
    new_config_dict = {
        "model_path": save_path,
        "tokenizer_path": base_cfg.get('tokenizer_path', ''),
        "vocab_size": base_cfg['vocab_size'],
        "hidden_dim": base_cfg['hidden_dim'],
        "num_layers": base_cfg['num_layers'],
        "num_heads": base_cfg['num_heads'],
        "num_kv_heads": base_cfg['num_kv_heads'],
        "initial_ffn_dim": new_ffn_dim  # 更新为新扩容后的维度
    }
    
    config_save_path = save_path.replace(".pt", "_config.json")
    with open(config_save_path, "w", encoding='utf-8') as f:
        json.dump(new_config_dict, f, indent=4)
    
    print(f"\n🎉 {Color.BOLD}全部流程圆满结束！{Color.RESET}")
    print(f"💾 完整抗遗忘模型已物理剥离并保存至: {Color.GREEN}{save_path}{Color.RESET}")
    print(f"📄 变异模型的新配置单已生成: {Color.GREEN}{config_save_path}{Color.RESET}\n")

if __name__ == "__main__":
    main()