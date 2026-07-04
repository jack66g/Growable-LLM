"""
GrowableLLM on HuggingFace backbone.

Replaces the hand-written Transformer in models.py with HF LlamaForCausalLM,
giving Flash Attention + torch.compile support while preserving the core
expand / HookLock / defrag mechanisms.

Usage:
    from models_hf import GrowableLlamaForCausalLM, GrowableLlamaConfig

    # From config dict (compatible with config.json model block)
    model = GrowableLlamaForCausalLM(config_dict)

    # From pretrained HF model ID
    model = GrowableLlamaForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")

    # Same API as GrowableLLM:
    model.expand_model(extra_dim=256, optimizer=optimizer)
    model.defrag(optimizer, fusion_data=input_ids)
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Union, Dict, Any

from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP


# ---------------------------------------------------------------------------
# Config adapter — maps ModelConfig-style keys to LlamaConfig keys
# ---------------------------------------------------------------------------

_CONFIG_KEY_MAP = {
    "hidden_dim": "hidden_size",
    "num_layers": "num_hidden_layers",
    "num_heads": "num_attention_heads",
    "num_kv_heads": "num_key_value_heads",
    "initial_ffn_dim": "intermediate_size",
    "max_seq_len": "max_position_embeddings",
}


def _to_llama_config(cfg: dict) -> LlamaConfig:
    mapped = {}
    for k, v in cfg.items():
        mapped[_CONFIG_KEY_MAP.get(k, k)] = v
    # SmolLM2 defaults that must be set
    mapped.setdefault("rope_theta", 100000)
    mapped.setdefault("rms_norm_eps", 1e-5)
    return LlamaConfig(**mapped)


# ---------------------------------------------------------------------------
# DynamicLlamaMLP — HF LlamaMLP with expand() and gradient-lock state
# ---------------------------------------------------------------------------

class DynamicLlamaMLP(LlamaMLP):
    """Same interface as LlamaMLP, but supports runtime dimension expansion.

    After expand(), the old hidden units are frozen via HookLock (gradient hooks)
    while the new units receive gradients normally.
    """

    def __init__(self, config):
        super().__init__(config)
        self.current_dim = config.intermediate_size
        self.locked_dim = 0

    def expand(self, extra_dim: int = 256):
        """Grow FFN width by *extra_dim* units, preserving existing weights.

        New rows/columns are Kaiming-uniform initialized (nn.Linear default).
        """
        old_dim = self.current_dim
        new_dim = old_dim + extra_dim
        device = self.gate_proj.weight.device
        dtype = self.gate_proj.weight.dtype
        bias = self.config.mlp_bias

        # ── gate_proj: [old_dim, hidden] → [new_dim, hidden] ──
        tmp_g = nn.Linear(self.hidden_size, new_dim, bias=bias).to(device, dtype)
        tmp_g.weight.data[:old_dim] = self.gate_proj.weight.data
        if bias:
            tmp_g.bias.data[:old_dim] = self.gate_proj.bias.data
        self.gate_proj = tmp_g

        # ── up_proj:   [old_dim, hidden] → [new_dim, hidden] ──
        tmp_u = nn.Linear(self.hidden_size, new_dim, bias=bias).to(device, dtype)
        tmp_u.weight.data[:old_dim] = self.up_proj.weight.data
        if bias:
            tmp_u.bias.data[:old_dim] = self.up_proj.bias.data
        self.up_proj = tmp_u

        # ── down_proj: [hidden, old_dim] → [hidden, new_dim] ──
        tmp_d = nn.Linear(new_dim, self.hidden_size, bias=bias).to(device, dtype)
        tmp_d.weight.data[:, :old_dim] = self.down_proj.weight.data
        if bias:
            # down_proj bias is [hidden_size], shape unchanged
            tmp_d.bias.data = self.down_proj.bias.data
        self.down_proj = tmp_d

        self.locked_dim = old_dim
        self.current_dim = new_dim


# ---------------------------------------------------------------------------
# GrowableLlamaForCausalLM — same API as GrowableLLM, HF-backed
# ---------------------------------------------------------------------------

class GrowableLlamaForCausalLM(nn.Module):
    """HF LlamaForCausalLM with GrowableLLM's expand / lock / defrag API.

    Two construction paths:
        >>> model = GrowableLlamaForCausalLM(config_dict)          # random init
        >>> model = GrowableLlamaForCausalLM.from_pretrained(...)  # pretrained
    """

    def __init__(self, config: Union[dict, LlamaConfig, LlamaForCausalLM]):
        super().__init__()
        # Accept a pre-built HF model directly (used by from_pretrained)
        if isinstance(config, LlamaForCausalLM):
            self.model = config
            self._replace_mlps()
            self.hooks = []
            return
        llama_cfg = _to_llama_config(config) if isinstance(config, dict) else config
        self.model = LlamaForCausalLM._from_config(llama_cfg)
        self._replace_mlps()
        self.hooks: list = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        """Load pretrained HF weights, then swap in DynamicLlamaMLP layers."""
        hf_model = LlamaForCausalLM.from_pretrained(model_id, **kwargs)
        # Passing the HF model to __init__ avoids creating a random-weight copy
        return cls(hf_model)

    # ── internal helpers ──────────────────────────────────────────────

    def _replace_mlps(self):
        """Replace each layer's LlamaMLP with a DynamicLlamaMLP, copying weights."""
        cfg = self.model.config
        for layer in self.model.model.layers:
            old_mlp = layer.mlp
            new_mlp = DynamicLlamaMLP(cfg)
            # Match the original model's dtype (typically bfloat16)
            new_mlp.to(old_mlp.gate_proj.weight.dtype)
            new_mlp.load_state_dict(old_mlp.state_dict())
            layer.mlp = new_mlp

    @property
    def _layers(self):
        return self.model.model.layers

    # ── forward ───────────────────────────────────────────────────────

    def forward(self, input_ids, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, labels=labels, **kwargs)
        return outputs.logits, outputs.loss

    # ── device / state_dict / train mode ──────────────────────────────

    def to(self, device, *args, **kwargs):
        self.model.to(device, *args, **kwargs)
        return self

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.model.load_state_dict(state_dict, *args, **kwargs)

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    # ── [Core 1] expand_model ────────────────────────────────────────

    def expand_model(self, extra_dim: int = 256, optimizer=None):
        """Expand all FFN layers, freeze old dimensions, sync optimizer."""
        for layer in self._layers:
            layer.mlp.expand(extra_dim)
        self.freeze_old_knowledge(global_lock=True)
        if optimizer is not None:
            self.sync_optimizer(optimizer)

    # ── [Core 2] HookLock ────────────────────────────────────────────

    def freeze_old_knowledge(self, global_lock: bool = False):
        """Register gradient hooks on all DynamicLlamaMLPs with locked_dim > 0.

        If *global_lock* is True, also freeze embedding / norm / attention params
        (called automatically by expand_model()).
        """
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

        if global_lock:
            self.model.model.embed_tokens.weight.requires_grad = False
            self.model.model.norm.weight.requires_grad = False
            for layer in self._layers:
                for p in layer.self_attn.parameters():
                    p.requires_grad = False
                layer.input_layernorm.weight.requires_grad = False
                layer.post_attention_layernorm.weight.requires_grad = False

        for layer in self._layers:
            mlp = layer.mlp
            ld = mlp.locked_dim
            if ld > 0:
                h1 = mlp.gate_proj.weight.register_hook(
                    partial(self._mask_grad_rows, locked=ld)
                )
                h2 = mlp.up_proj.weight.register_hook(
                    partial(self._mask_grad_rows, locked=ld)
                )
                h3 = mlp.down_proj.weight.register_hook(
                    partial(self._mask_grad_cols, locked=ld)
                )
                self.hooks.extend([h1, h2, h3])

    @staticmethod
    def _mask_grad_rows(grad, locked):
        grad[:locked] = 0.0
        return grad

    @staticmethod
    def _mask_grad_cols(grad, locked):
        grad[:, :locked] = 0.0
        return grad

    # ── [Core 3] sync_optimizer ──────────────────────────────────────

    def sync_optimizer(self, optimizer):
        """Remove stale parameter keys from optimizer state and add new trainable params."""
        model_param_ids = set(id(p) for p in self.parameters() if p.requires_grad)

        stale = [p for p in optimizer.state.keys() if id(p) not in model_param_ids]
        for p in stale:
            del optimizer.state[p]

        for group in optimizer.param_groups:
            group["params"] = [
                p for p in group["params"] if id(p) in model_param_ids
            ]

        tracked = set(id(p) for group in optimizer.param_groups for p in group["params"])
        new_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in tracked
        ]
        if new_params:
            optimizer.add_param_group({"params": new_params})

    # ── [Core 4] defrag ────────────────────────────────────────

    def defrag(self, optimizer, fusion_data: Optional[torch.Tensor] = None, target_replay_size: int = 32):
        """Unlock last 6 layers' attention + final norm, replay *fusion_data*.

        This aligns the newly added hidden units with existing attention distributions.
        """
        if fusion_data is None:
            print("⚠️  No fusion data provided, skipping defrag.")
            return

        orig_requires_grad = {id(p): p.requires_grad for p in self.parameters()}

        try:
            # ── 1. Freeze everything ──
            for p in self.parameters():
                p.requires_grad = False

            # ── 2. Unlock last 6 layers' self-attention ──
            unlock_start = max(0, len(self._layers) - 6)
            for i in range(unlock_start, len(self._layers)):
                layer = self._layers[i]
                for name, p in layer.named_parameters():
                    if "self_attn" in name:
                        p.requires_grad = True
                layer.input_layernorm.weight.requires_grad = True
                layer.post_attention_layernorm.weight.requires_grad = True

            # ── 3. Unlock final norm ──
            self.model.model.norm.weight.requires_grad = True

            # ── 4. Lower LR temporarily ──
            orig_lrs = [pg["lr"] for pg in optimizer.param_groups]
            for pg in optimizer.param_groups:
                pg["lr"] = 1e-5

            self.sync_optimizer(optimizer)

            # ── 5. Replay fusion data ──
            for _ in range(target_replay_size):
                optimizer.zero_grad()
                _, loss = self(fusion_data, labels=fusion_data)
                if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()

            # ── 6. Restore LRs ──
            for pg, lr in zip(optimizer.param_groups, orig_lrs):
                pg["lr"] = lr

        finally:
            # ── 7. Restore original requires_grad ──
            for p in self.parameters():
                if id(p) in orig_requires_grad:
                    p.requires_grad = orig_requires_grad[id(p)]
            self.sync_optimizer(optimizer)

    # ── generate (delegates to HF generate) ───────────────────────────

    @torch.no_grad()
    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids, **kwargs)