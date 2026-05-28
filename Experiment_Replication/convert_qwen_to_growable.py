import torch
from transformers import AutoModelForCausalLM
from model import GrowableLLM, ModelConfig

def main():
    print("⏳ 正在从 HuggingFace 提取 Qwen1.5-0.5B 官方大脑矩阵...")
    # 自动下载或加载本地缓存的 Qwen 模型
    qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-0.5B", 
        torch_dtype=torch.float32, # 为了保证兼容性，先用 fp32 提取
        trust_remote_code=True
    )
    
    qwen_state_dict = qwen_model.state_dict()
    growable_state_dict = {}

    print("🧬 开始执行神经元映射与器官移植...")

    # 1. 映射词表和输出头
    growable_state_dict["embed.weight"] = qwen_state_dict["model.embed_tokens.weight"]
    growable_state_dict["lm_head.weight"] = qwen_state_dict["lm_head.weight"]
    growable_state_dict["norm.weight"] = qwen_state_dict["model.norm.weight"]

    # 2. 映射 24 层 Transformer 块
    for i in range(24): # Qwen1.5-0.5B 有 24 层
        prefix_qwen = f"model.layers.{i}."
        prefix_grow = f"blocks.{i}."

        # Attention 层映射
        growable_state_dict[f"{prefix_grow}attn.q_proj.weight"] = qwen_state_dict[f"{prefix_qwen}self_attn.q_proj.weight"]
        growable_state_dict[f"{prefix_grow}attn.k_proj.weight"] = qwen_state_dict[f"{prefix_qwen}self_attn.k_proj.weight"]
        growable_state_dict[f"{prefix_grow}attn.v_proj.weight"] = qwen_state_dict[f"{prefix_qwen}self_attn.v_proj.weight"]
        growable_state_dict[f"{prefix_grow}attn.o_proj.weight"] = qwen_state_dict[f"{prefix_qwen}self_attn.o_proj.weight"]
        
        # FFN 层映射 (你的 DynamicSwiGLU)
        growable_state_dict[f"{prefix_grow}ffn.gate_proj.weight"] = qwen_state_dict[f"{prefix_qwen}mlp.gate_proj.weight"]
        growable_state_dict[f"{prefix_grow}ffn.up_proj.weight"] = qwen_state_dict[f"{prefix_qwen}mlp.up_proj.weight"]
        growable_state_dict[f"{prefix_grow}ffn.down_proj.weight"] = qwen_state_dict[f"{prefix_qwen}mlp.down_proj.weight"]

        # LayerNorm 映射
        growable_state_dict[f"{prefix_grow}attn_norm.weight"] = qwen_state_dict[f"{prefix_qwen}input_layernorm.weight"]
        growable_state_dict[f"{prefix_grow}ffn_norm.weight"] = qwen_state_dict[f"{prefix_qwen}post_attention_layernorm.weight"]

    print("✅ 矩阵重铸完成！正在验证形状契合度...")
    
    # 3. 实例化你的架构并加载
    config = ModelConfig()
    my_model = GrowableLLM(config)
    
    # strict=False 允许跳过不需要加载的数学常量
    missing, unexpected = my_model.load_state_dict(growable_state_dict, strict=False)
    
    # 过滤掉 RoPE 的 cos/sin 缓存（这是正常的，模型会自己生成）
    real_missing = [k for k in missing if "rope" not in k]
    
    if len(real_missing) == 0 and len(unexpected) == 0:
        print("🎉 完美契合！所有核心突触全部移植成功（RoPE 常量缓存已由模型自动生成）！")
    else:
        if len(real_missing) > 0:
            print(f"⚠️ 真正缺失的神经元: {real_missing}")
        if len(unexpected) > 0:
            print(f"⚠️ 多余的神经元: {unexpected}")

    # 4. 保存为你的专属初始基座
    save_path = "growable_qwen_base.pth"
    torch.save(growable_state_dict, save_path)
    print(f"\n🏆 移植成功！属于你的全能初始基座已保存至: {save_path}")
    print("接下来，你可以直接用它去拓展你的【高情商聊天】或【中医脑区】了！")

if __name__ == "__main__":
    main()