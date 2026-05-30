# ============================================================
# SmolLM2-360M-Instruct -> GrowableLLM 权重提取与转换脚本
# ============================================================

import torch
from transformers import AutoModelForCausalLM
from model import GrowableLLM, ModelConfig  # 导入你的模型架构

def convert_hf_to_growable_llm():
    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    print(f"📥 正在从 Hugging Face 下载并加载 {model_id} ...")
    
    # 1. 加载官方模型 (使用 bfloat16 节省内存，360M 仅需约 700MB)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    hf_state_dict = hf_model.state_dict()
    
    # 2. 初始化你的 GrowableLLM 架构
    # 这些是 SmolLM2-360M 的真实物理参数
    config = ModelConfig(
        vocab_size=49152,  # 占位，马上会动态修正
        hidden_dim=960,
        num_layers=32,
        num_heads=15,
        num_kv_heads=5,
        initial_ffn_dim=2560,
        rope_theta=100000,
    )
    
    # 动态获取实际词表大小 (防止大厂词表有 Padding)
    actual_vocab_size = hf_state_dict["model.embed_tokens.weight"].shape[0]
    config.vocab_size = actual_vocab_size
    print(f"📐 探测到实际词表大小: {actual_vocab_size}")
    
    print("🚀 初始化 GrowableLLM 基座...")
    custom_model = GrowableLLM(config)
    
    # 3. 开始外科手术式映射权重
    print("🔄 开始映射权重矩阵...")
    custom_state_dict = custom_model.state_dict()
    
    mapping_count = 0
    for name, param in hf_state_dict.items():
        # 词表和最终的 Norm
        if name == "model.embed_tokens.weight":
            custom_state_dict["embed.weight"].copy_(param)
            mapping_count += 1
        elif name == "model.norm.weight":
            custom_state_dict["norm.weight"].copy_(param)
            mapping_count += 1
        elif name == "lm_head.weight":
            custom_state_dict["lm_head.weight"].copy_(param)
            mapping_count += 1
            
        # 映射 Transformer Blocks (32 层循环映射)
        elif name.startswith("model.layers."):
            parts = name.split(".")
            layer_idx = parts[2]
            
            # Norms
            if "input_layernorm.weight" in name:
                custom_state_dict[f"blocks.{layer_idx}.attn_norm.weight"].copy_(param)
                mapping_count += 1
            elif "post_attention_layernorm.weight" in name:
                custom_state_dict[f"blocks.{layer_idx}.ffn_norm.weight"].copy_(param)
                mapping_count += 1
                
            # Attention (GQA 架构映射)
            elif "self_attn.q_proj.weight" in name:
                custom_state_dict[f"blocks.{layer_idx}.attn.q_proj.weight"].copy_(param)
                mapping_count += 1
            elif "self_attn.k_proj.weight" in name:
                custom_state_dict[f"blocks.{layer_idx}.attn.k_proj.weight"].copy_(param)
                mapping_count += 1
            elif "self_attn.v_proj.weight" in name:
                custom_state_dict[f"blocks.{layer_idx}.attn.v_proj.weight"].copy_(param)
                mapping_count += 1
            elif "self_attn.o_proj.weight" in name:
                custom_state_dict[f"blocks.{layer_idx}.attn.o_proj.weight"].copy_(param)
                mapping_count += 1
                
            # FFN (DynamicSwiGLU 初始态映射)
            elif "mlp.gate_proj.weight" in name:
                custom_state_dict[f"blocks.{layer_idx}.ffn.gate_proj.weight"].copy_(param)
                mapping_count += 1
            elif "mlp.up_proj.weight" in name:
                custom_state_dict[f"blocks.{layer_idx}.ffn.up_proj.weight"].copy_(param)
                mapping_count += 1
            elif "mlp.down_proj.weight" in name:
                custom_state_dict[f"blocks.{layer_idx}.ffn.down_proj.weight"].copy_(param)
                mapping_count += 1

    # 如果 SmolLM2 的权重绑定了 Embed 和 lm_head，我们手动同步一下
    if "lm_head.weight" not in hf_state_dict:
        custom_state_dict["lm_head.weight"].copy_(custom_state_dict["embed.weight"])

    custom_model.load_state_dict(custom_state_dict)
    print(f"✅ 成功映射 {mapping_count} 个张量！(架构完美咬合)")
    
    # 4. 保存为纯净的本地 PyTorch 模型
    save_path = "smollm2_360m_growable.pt"
    torch.save(custom_model.state_dict(), save_path)
    print(f"💾 权重已物理剥离，并保存至: {save_path}")
    print("🔥 接下来你可以直接加载这个 .pt 文件，彻底告别 HuggingFace 约束！")

if __name__ == "__main__":
    convert_hf_to_growable_llm()