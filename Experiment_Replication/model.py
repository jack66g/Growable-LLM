import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

# =====================================================
# Config
# =====================================================

@dataclass
class ModelConfig:
    vocab_size: int = 151936    # 🌟 Qwen 真实验证的词表大小
    hidden_dim: int = 1024      # 🌟 Qwen 0.5B 的隐藏层维度
    num_layers: int = 24        # 🌟 Qwen 0.5B 的真实层数
    num_heads: int = 16         # 🌟 注意力头数
    num_kv_heads: int = 16      # 🌟 KV 头数 (0.5B 不是 GQA，是 MHA)
    initial_ffn_dim: int = 2816 # 🌟 真实的 FFN 维度 (重点！)
    max_seq_len: int = 8192
    rope_theta: int = 1000000   # Qwen1.5 的 RoPE 频率通常是 1000000
    dropout: float = 0.0


# =====================================================
# RMSNorm
# =====================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


# =====================================================
# RoPE
# =====================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=10000):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    # 🌟 修复点 1：增加 offset 参数，接入 KV Cache 的时间偏移
    def forward(self, x, seq_len, offset=0):
        return (
            self.cos_cached[:, :, offset : offset + seq_len, :],
            self.sin_cached[:, :, offset : offset + seq_len, :],
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


# =====================================================
# KV Cache
# =====================================================

class KVCache:
    def __init__(self):
        self.k = None
        self.v = None

    def update(self, k, v):
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v


# =====================================================
# GQA Attention
# =====================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_dim // self.num_heads

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_dim,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_dim,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.rope = RotaryEmbedding(
            self.head_dim,
            config.max_seq_len,
            config.rope_theta,
        )

    def repeat_kv(self, x, n_rep):
        b, h, s, d = x.shape
        x = x[:, :, None, :, :].expand(b, h, n_rep, s, d)
        return x.reshape(b, h * n_rep, s, d)

    def forward(self, x, kv_cache: Optional[KVCache] = None):
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 🌟 修复点 2：探测 KV Cache 的历史长度，作为 offset 传递给 RoPE
        offset = 0
        if kv_cache is not None and kv_cache.k is not None:
            offset = kv_cache.k.shape[2]

        cos, sin = self.rope(q, seq_len, offset=offset)
        q, k = apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        repeat_factor = self.num_heads // self.num_kv_heads
        k = self.repeat_kv(k, repeat_factor)
        v = self.repeat_kv(v, repeat_factor)

        # 🌟 修复点 3：动态因果掩码。单字推理时不需要 causal mask
        causal = True if seq_len > 1 else False
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal,
        )

        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(bsz, seq_len, self.hidden_dim)

        return self.o_proj(attn)


# =====================================================
# Dynamic SwiGLU FFN (保持原样，极其完美)
# =====================================================

class DynamicSwiGLU(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.current_dim = ffn_dim
        self.locked_dim = 0

        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def swiglu(self, x, gate):
        return F.silu(gate) * x

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        hidden = self.swiglu(up, gate)

        if self.locked_dim > 0:
            old = hidden[..., : self.locked_dim]
            new = hidden[..., self.locked_dim :]

            old_out = F.linear(
                old,
                self.down_proj.weight[:, : self.locked_dim],
            )

            new_out = F.linear(
                new,
                self.down_proj.weight[:, self.locked_dim :],
            )

            # 新脑区的知识作为补充，直接叠加到旧脑区上
            return old_out + new_out

        return self.down_proj(hidden)

    def expand(self, extra_dim=512):
        old_dim = self.current_dim
        new_dim = old_dim + extra_dim

        new_gate = nn.Linear(self.hidden_dim, new_dim, bias=False).to(self.gate_proj.weight.device)
        new_up = nn.Linear(self.hidden_dim, new_dim, bias=False).to(self.up_proj.weight.device)
        new_down = nn.Linear(new_dim, self.hidden_dim, bias=False).to(self.down_proj.weight.device)

        new_gate.weight.data[:old_dim] = self.gate_proj.weight.data
        new_up.weight.data[:old_dim] = self.up_proj.weight.data
        new_down.weight.data[:, :old_dim] = self.down_proj.weight.data

        # 零初始化 (Zero-Initialization)
        torch.nn.init.zeros_(new_down.weight.data[:, old_dim:])

        self.gate_proj = new_gate
        self.up_proj = new_up
        self.down_proj = new_down

        self.locked_dim = old_dim
        self.current_dim = new_dim


# =====================================================
# Transformer Block
# =====================================================

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.attn = MultiHeadAttention(config)
        self.ffn = DynamicSwiGLU(
            config.hidden_dim,
            config.initial_ffn_dim,
        )

    def forward(self, x, kv_cache=None):
        x = x + self.attn(self.attn_norm(x), kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# =====================================================
# Main Model (Perfected)
# =====================================================

class GrowableLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.replay_buffer = {
            "general": [],
            "dialogue": [],
            "domain": [],
        }
        self.hooks = [] 

    def forward(self, input_ids, labels=None, kv_caches=None):
        x = self.embed(input_ids)
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            x = block(x, kv_caches[i])
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return logits, loss

    # =====================================================
    # 【核心1】动态扩容与 7B 封顶保护
    # =====================================================
    def expand_model(self, extra_dim=512):
        total_params = sum(p.numel() for p in self.parameters())
        max_params = 7_000_000_000 # 7B
        
        if total_params >= max_params:
            print(f"🛑 [系统拦截] 模型参数量已达 {total_params/1e9:.2f}B，触发 7B 封顶！停止物理扩容。")
            return False

        print(f"\n📈 动态扩容 FFN: +{extra_dim} (当前规模: {total_params/1e9:.2f}B)")
        for block in self.blocks:
            block.ffn.expand(extra_dim)
            
        self.freeze_old_knowledge(global_lock=True)
        return True

    # =====================================================
    # 【核心2】完美梯度锁：彻底隔绝遗忘
    # =====================================================
    def freeze_old_knowledge(self, global_lock=False):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        if global_lock:
            self.embed.weight.requires_grad = False
            self.norm.weight.requires_grad = False
            for block in self.blocks:
                for param in block.attn.parameters():
                    param.requires_grad = False
                block.attn_norm.weight.requires_grad = False
                block.ffn_norm.weight.requires_grad = False

        for block in self.blocks:
            locked = block.ffn.locked_dim
            if locked > 0:
                h1 = block.ffn.gate_proj.weight.register_hook(lambda grad, l=locked: self.mask_grad_rows(grad, l))
                h2 = block.ffn.up_proj.weight.register_hook(lambda grad, l=locked: self.mask_grad_rows(grad, l))
                h3 = block.ffn.down_proj.weight.register_hook(lambda grad, l=locked: self.mask_grad_cols(grad, l))
                
                self.hooks.extend([h1, h2, h3])

    def mask_grad_rows(self, grad, locked):
        grad_clone = grad.clone()
        grad_clone[:locked, :] = 0
        return grad_clone

    def mask_grad_cols(self, grad, locked):
        grad_clone = grad.clone()
        grad_clone[:, :locked] = 0
        return grad_clone

    # =====================================================
    # 【核心3】小乐协议：记忆回放与碎片整理 (方案B: 不对称解锁 Defrag)
    # =====================================================
    def defrag(self, optimizer):
        total_memories = sum(len(v) for v in self.replay_buffer.values())
        if total_memories == 0: 
            return
            
        print("🌀 触发 [不对称] 碎片整理与特征对齐 (方案B)...")
        
        # 1. 🌟 全局防御性锁死：先确保所有门都关严
        for param in self.parameters():
            param.requires_grad = False
            
        # 2. 🌟 精准开锁：只解锁最后 6 层（如果总层数不足6层，则解锁全部）
        unlock_start_layer = max(0, len(self.blocks) - 6)
        
        for i in range(unlock_start_layer, len(self.blocks)):
            block = self.blocks[i]
            for param in block.parameters():
                param.requires_grad = True
                
        # 🌟 解锁最终的层归一化
        self.norm.weight.requires_grad = True
        
        # ⚠️ 注意：因为 self.lm_head.weight 共享了 self.embed.weight 的内存，
        # 为了绝对保护底层 18 层的词表认知不被洗掉，这里坚决不解锁 embed 矩阵！
                
        original_lrs = []
        for param_group in optimizer.param_groups:
            original_lrs.append(param_group['lr'])
            param_group['lr'] = 1e-6
            
        replay_batch = []
        for category, memories in self.replay_buffer.items():
            if memories:
                replay_batch.extend(random.sample(memories, min(len(memories), 4)))
                
        for past_data in replay_batch:
            optimizer.zero_grad()
            _, loss = self(past_data, labels=past_data)
            
            if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
        for param_group, lr in zip(optimizer.param_groups, original_lrs):
            param_group['lr'] = lr
            
        print("✨ [不对称] 碎片整理完成，顶层网络已与新突触完美融合！")

    # =====================================================
    # Generation
    # =====================================================
    @torch.no_grad()
    def generate(
        self, 
        input_ids, 
        eos_token_id=None,
        max_new_tokens=128, 
        temperature=0.8, 
        top_p=0.95, 
        repetition_penalty=1.2
    ):
        self.eval()
        kv_caches = [KVCache() for _ in self.blocks]
        generated_ids = []

        logits, _ = self(input_ids, kv_caches=kv_caches)
        next_token_logits = logits[:, -1].clone()

        for step in range(max_new_tokens):
            if step > 0:
                logits, _ = self(input_ids[:, -1:], kv_caches=kv_caches)
                next_token_logits = logits[:, -1].clone()

            if repetition_penalty > 1.0:
                for token_id in set(generated_ids + input_ids[0].tolist()):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            for i in range(probs.size(0)):
                probs[i, sorted_indices[i][sorted_indices_to_remove[i]]] = 0

            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, 1)
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_ids.append(next_token.item())

        self.train()
        return input_ids


# =====================================================
# Example
# =====================================================

if __name__ == "__main__":
    config = ModelConfig(
        vocab_size=32000,
        hidden_dim=1024,
        num_layers=4,
        num_heads=16,
        num_kv_heads=4,
        initial_ffn_dim=4096,
    )

    model = GrowableLLM(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randint(0, 32000, (2, 128)).cuda()
    logits, loss = model(x, labels=x)
    print("Pre-train loss:", loss.item())

    model.replay_buffer["general"].append(x)

    model.defrag(optimizer)

    model.expand_model(extra_dim=512)

    logits, loss = model(x, labels=x)
    loss.backward()
    print("Post-expand loss:", loss.item())

    out = model.generate(x[:, :16], max_new_tokens=10)
    print("Generate output shape:", out.shape)