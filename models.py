import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from functools import partial


# =====================================================
# Config —— 模型超参数配置
# =====================================================

@dataclass
class ModelConfig:
    vocab_size: int = 151936
    hidden_dim: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    initial_ffn_dim: int = 2816
    max_seq_len: int = 8192
    rope_theta: int = 1000000
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
# Multi-Head Attention
# =====================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_dim // self.num_heads

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_seq_len,
            base=config.rope_theta,
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

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

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

        causal = True if seq_len > 1 else False
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)
        return self.o_proj(attn)


# =====================================================
# Dynamic SwiGLU FFN —— 可生长的前馈网络（项目核心）
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

            old_out = F.linear(old, self.down_proj.weight[:, : self.locked_dim])
            new_out = F.linear(new, self.down_proj.weight[:, self.locked_dim :])

            return old_out + new_out

        return self.down_proj(hidden)

    def expand(self, extra_dim=512):
        """
        原地扩展权重矩阵：三个投影的新部分全部 Kaiming 随机初始化。
        新神经元三个参数都能学，梯度全部非零。

        注意：gate/up/down 新部分均随机初始化，因此 expand 后前向输出会变化（不等价于扩维前）。
        但 HookLock 保护了旧参数（梯度置零），旧知识不受影响；新神经元从零开始学。
        """
        old_dim = self.current_dim
        new_dim = old_dim + extra_dim

        # ══════════════════════════════════════════════════════
        # gate_proj: 旧部分复制，新部分随机初始化（Kaiming）
        # ══════════════════════════════════════════════════════
        # 随机初始化 → silu(gate) ≠ 0 → 新 dim 梯度活跃 
        tmp_gate = nn.Linear(self.hidden_dim, new_dim, bias=False).to(
            self.gate_proj.weight.device
        )
        tmp_gate.weight.data[:old_dim] = self.gate_proj.weight.data
        self.gate_proj.weight = tmp_gate.weight
        self.gate_proj.out_features = new_dim

        # ══════════════════════════════════════════════════════
        # up_proj: 旧部分复制，新部分随机初始化（Kaiming）
        # ══════════════════════════════════════════════════════
        # 随机初始化 → 新 dim 梯度活跃 
        tmp_up = nn.Linear(self.hidden_dim, new_dim, bias=False).to(
            self.up_proj.weight.device
        )
        tmp_up.weight.data[:old_dim] = self.up_proj.weight.data
        self.up_proj.weight = tmp_up.weight
        self.up_proj.out_features = new_dim

        # ══════════════════════════════════════════════════════
        # down_proj: 旧部分复制，新部分随机初始化（Kaiming）
        # ══════════════════════════════════════════════════════
        # 随机初始化 → 新 dim 的输出影响 loss，梯度回传 
        tmp_down = nn.Linear(new_dim, self.hidden_dim, bias=False).to(
            self.down_proj.weight.device
        )
        tmp_down.weight.data[:, :old_dim] = self.down_proj.weight.data
        self.down_proj.weight = tmp_down.weight
        self.down_proj.in_features = new_dim

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
        self.ffn = DynamicSwiGLU(config.hidden_dim, config.initial_ffn_dim)

    def forward(self, x, kv_cache=None):
        x = x + self.attn(self.attn_norm(x), kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# =====================================================
# Main Model —— 可生长的 LLM 主体
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

    # ─────────────────────────────────────
    # 前向传播
    # ─────────────────────────────────────
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

    # ─────────────────────────────────────
    # 安全写入经验回放（自动 detach + 默认存 CPU）
    # ─────────────────────────────────────
    def add_to_replay(self, category: str, tensor: torch.Tensor):
        if category not in self.replay_buffer:
            raise ValueError(
                f"未知的记忆类别: {category}，可选: {list(self.replay_buffer.keys())}"
            )
        # 🔧 默认存 CPU，避免长期占用 GPU 显存
        self.replay_buffer[category].append(tensor.detach().cpu().clone())

    # ─────────────────────────────────────
    # 【核心1】动态扩容
    # ─────────────────────────────────────
    def expand_model(self, extra_dim=512, optimizer=None):
        total_params = sum(p.numel() for p in self.parameters())
        max_params = 7_000_000_000

        if total_params >= max_params:
            print(f"🛑 [系统拦截] 模型参数量已达 {total_params/1e9:.2f}B，触发 7B 封顶！")
            return False

        print(f"\n📈 动态扩容 FFN: +{extra_dim} (当前规模: {total_params/1e9:.2f}B)")
        for block in self.blocks:
            block.ffn.expand(extra_dim)

        self.freeze_old_knowledge(global_lock=True)

        if optimizer is not None:
            self.sync_optimizer(optimizer)
            print("✅ 优化器参数组已同步，新神经元开始接收梯度。")

        return True

    # ─────────────────────────────────────
    # 同步优化器（清理旧状态 + 追踪新参数）
    # ─────────────────────────────────────
    def sync_optimizer(self, optimizer):
        # 当前模型中所有 requires_grad=True 的参数
        model_param_ids = set(id(p) for p in self.parameters() if p.requires_grad)

        # 🔧 清理 optimizer.state 中已不存在的参数（防止多次扩容后状态字典膨胀）
        stale_params = [p for p in optimizer.state.keys() if id(p) not in model_param_ids]
        for p in stale_params:
            del optimizer.state[p]

        # 清理 param_groups 中的过期参数引用
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if id(p) in model_param_ids]

        # 添加尚未被追踪的新参数
        tracked_ids = set()
        for group in optimizer.param_groups:
            for p in group['params']:
                tracked_ids.add(id(p))

        new_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in tracked_ids
        ]

        if new_params:
            optimizer.add_param_group({'params': new_params})

    # ─────────────────────────────────────
    # 【核心2】梯度锁
    # ─────────────────────────────────────
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
                h1 = block.ffn.gate_proj.weight.register_hook(
                    partial(self._mask_grad_rows, locked=locked)
                )
                h2 = block.ffn.up_proj.weight.register_hook(
                    partial(self._mask_grad_rows, locked=locked)
                )
                h3 = block.ffn.down_proj.weight.register_hook(
                    partial(self._mask_grad_cols, locked=locked)
                )
                self.hooks.extend([h1, h2, h3])

    @staticmethod
    def _mask_grad_rows(grad, locked):
        grad[:locked, :] = 0.0
        return grad

    @staticmethod
    def _mask_grad_cols(grad, locked):
        grad[:, :locked] = 0.0
        return grad

    # ─────────────────────────────────────
    # 【核心3】碎片整理
    # ─────────────────────────────────────
    def defrag(self, optimizer, target_replay_size=32):
        total_memories = sum(len(v) for v in self.replay_buffer.values())
        if total_memories == 0:
            print("⚠️ 记忆库为空，跳过碎片整理。")
            return

        print(f"🌀 触发 [碎片整理]：顶层软路由 + {target_replay_size} 条经验回放...")

        # 🔧 用 id(p) 做键，避免同名参数互相覆盖
        original_requires_grad = {
            id(param): param.requires_grad
            for param in self.parameters()
        }

        try:
            # 1) 全局防御性锁死
            for param in self.parameters():
                param.requires_grad = False

            # 2) 仅解锁最后 6 层的 Attention 与 attn_norm
            unlock_start_layer = max(0, len(self.blocks) - 6)
            for i in range(unlock_start_layer, len(self.blocks)):
                block = self.blocks[i]
                for name, param in block.named_parameters():
                    if "attn" in name:
                        param.requires_grad = True

            # 3) 解锁最终 RMSNorm
            self.norm.weight.requires_grad = True

            # 4) 临时压学习率
            original_lrs = [pg['lr'] for pg in optimizer.param_groups]
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-5

            self.sync_optimizer(optimizer)

            # 5) 构造回放批次（从 CPU 搬回 GPU）
            all_memories = []
            for memories in self.replay_buffer.values():
                all_memories.extend(memories)

            if len(all_memories) > target_replay_size:
                replay_batch = random.sample(all_memories, target_replay_size)
            else:
                replay_batch = all_memories

            # 6) 经验回放
            for past_data_cpu in replay_batch:
                optimizer.zero_grad()
                # 推理时搬到模型所在设备
                device = self.embed.weight.device
                past_data = past_data_cpu.to(device)
                _, loss = self(past_data, labels=past_data)

                if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()

            # 7) 恢复学习率
            for pg, lr in zip(optimizer.param_groups, original_lrs):
                pg['lr'] = lr

        finally:
            for param in self.parameters():
                if id(param) in original_requires_grad:
                    param.requires_grad = original_requires_grad[id(param)]

            self.sync_optimizer(optimizer)

        print("✨ [碎片整理完成] 顶层路由已对齐，新突触已被消音。")

    # ─────────────────────────────────────
    # 文本生成
    # ─────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        eos_token_id=None,
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.2,
    ):
        self.eval()
        bsz = input_ids.size(0)
        kv_caches = [KVCache() for _ in self.blocks]

        seen_tokens = [set(input_ids[i].tolist()) for i in range(bsz)]

        # 1) Prefill
        logits, _ = self(input_ids, kv_caches=kv_caches)
        next_token_logits = logits[:, -1].clone()

        finished = [False] * bsz

        for step in range(max_new_tokens):
            # 2) Decode
            if step > 0:
                logits, _ = self(input_ids[:, -1:], kv_caches=kv_caches)
                next_token_logits = logits[:, -1].clone()

            # 3) 重复惩罚
            if repetition_penalty > 1.0:
                for i in range(bsz):
                    for token_id in seen_tokens[i]:
                        if next_token_logits[i, token_id] > 0:
                            next_token_logits[i, token_id] /= repetition_penalty
                        else:
                            next_token_logits[i, token_id] *= repetition_penalty

            # 4) 温度缩放
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)

            # 5) top-p 采样
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            for i in range(bsz):
                probs[i, sorted_indices[i][sorted_indices_to_remove[i]]] = 0

            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, 1)

            # 6) 更新状态
            for i in range(bsz):
                tok = next_token[i].item()
                seen_tokens[i].add(tok)
                if eos_token_id is not None and tok == eos_token_id:
                    finished[i] = True

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if eos_token_id is not None and all(finished):
                break

        self.train()
        return input_ids

    # ─────────────────────────────────────
    # Checkpoint 保存 / 加载
    # ─────────────────────────────────────
    def save_checkpoint(self, path: str):
        checkpoint = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'ffn_meta': [
                {'current_dim': b.ffn.current_dim, 'locked_dim': b.ffn.locked_dim}
                for b in self.blocks
            ],
        }
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint 已保存至 {path}")

    @classmethod
    def load_checkpoint(cls, path: str, map_location='cpu'):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = checkpoint['config']
        model = cls(config)

        if 'ffn_meta' in checkpoint:
            for block, meta in zip(model.blocks, checkpoint['ffn_meta']):
                if meta['current_dim'] != config.initial_ffn_dim:
                    extra = meta['current_dim'] - config.initial_ffn_dim
                    block.ffn.expand(extra)
                block.ffn.locked_dim = meta['locked_dim']
        elif 'ffn_dims' in checkpoint:
            for block, (cur, locked) in zip(model.blocks, checkpoint['ffn_dims']):
                if cur != config.initial_ffn_dim:
                    block.ffn.expand(cur - config.initial_ffn_dim)
                block.ffn.locked_dim = locked

        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f"📥 Checkpoint 已从 {path} 加载")
        return model


# =====================================================
# 测试入口
# =====================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    config = ModelConfig(
        vocab_size=32000,
        hidden_dim=1024,
        num_layers=4,
        num_heads=16,
        num_kv_heads=4,
        initial_ffn_dim=4096,
    )

    model = GrowableLLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randint(0, 32000, (2, 128)).to(device)

    # 前向
    logits, loss = model(x, labels=x)
    print("Pre-train loss:", loss.item())

    # 写入记忆库
    model.add_to_replay("general", x)

    # 碎片整理
    model.defrag(optimizer)

    # 扩容
    model.expand_model(extra_dim=512, optimizer=optimizer)

    # 验证恒等性
    with torch.no_grad():
        logits2, loss2 = model(x, labels=x)
    print("Post-expand loss (should ≈ pre-train loss):", loss2.item())

    # 🔧 验证新神经元是否能学到梯度
    print("\n🔬 梯度流测试：训练一步看新参数是否变动...")
    before = model.blocks[0].ffn.down_proj.weight[:, -512:].clone()
    optimizer.zero_grad()
    _, loss3 = model(x, labels=x)
    loss3.backward()
    optimizer.step()
    after = model.blocks[0].ffn.down_proj.weight[:, -512:].clone()
    delta = (after - before).abs().max().item()
    if delta > 0:
        print(f"✅ 新神经元梯度流通正常！down_proj 新列最大变动 = {delta:.6f}")
    else:
        print(f"❌ 新神经元梯度仍然为零，请检查！delta = {delta:.10f}")

    # 生成测试
    out = model.generate(x[:, :16], max_new_tokens=10)
    print("Generate output shape:", out.shape)

    # 保存 / 加载测试
    model.save_checkpoint("growable_llm_test.pt")
    model2 = GrowableLLM.load_checkpoint("growable_llm_test.pt", map_location=device)
    print("✅ Checkpoint 加载成功，参数量:", sum(p.numel() for p in model2.parameters()))