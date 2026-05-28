import os
import torch
import types
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import GrowableLLM, ModelConfig

# =====================================================
# [配置区：双脑区唤醒 + 年轮物理路由]
# =====================================================
WEIGHT_PATH = "growable_med_expert_epoch3.pth" # 🌟 第二阶段老中医出关权重
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🌟 物理路由坐标字典 (年轮累加法：纯净底座2816 -> 聊天层3072 -> 中医层3200)
BRAIN_CUTOFFS = {
    "chat": 3072,  # 开放到聊天层边界 (物理屏蔽后面的中医层突触，防止性格串味)
    "med":  3200   # 开放到中医层边界 (全开，调用全部 3200 维算力)
}

# 🌟 双核心脑区密钥 (System Prompts)
PROMPTS = {
    "chat": "你是一个高情商、懂人情世故、乐于助人的AI助手。",
    "med": "你是一位精通传统中医（TCM）的专家。你能根据患者的症状，准确辨证论治，并给出严谨的中药或方剂建议。"
}

# 推理参数 (针对医疗场景稍微调低温度，防幻觉)
TEMPERATURE = 0.6
TOP_P = 0.85
REP_PENALTY = 1.15
MAX_NEW_TOKENS = 512

# =====================================================
# [黑魔法：动态注入年轮物理路由器]
# =====================================================
def apply_brain_routing(model, persona):
    """
    通过 Monkey Patch 动态修改模型内部的 SwiGLU 激活函数。
    实现：保留历史生长的脑区，绝对屏蔽未来的脑区噪音！
    """
    cutoff_dim = BRAIN_CUTOFFS.get(persona)
    
    def masked_swiglu(self, x, gate):
        # 原始的激活计算
        hidden = F.silu(gate) * x
        
        # 🛡️ 年轮路由切片
        if cutoff_dim is not None:
            # 1. 创建一个与 hidden 形状相同的全1掩码
            mask = torch.ones_like(hidden)
            
            # 2. 一刀切断 cutoff_dim 之后的所有【未来外挂维度】，强制物理断电！
            mask[..., cutoff_dim:] = 0.0
            
            # 3. 执行物理截断
            hidden = hidden * mask
            
        return hidden

    # 遍历所有 Transformer 层，强行篡改它的 swiglu 内存指针！
    for block in model.blocks:
        block.ffn.swiglu = types.MethodType(masked_swiglu, block.ffn)


# =====================================================
# [核心生成逻辑：带护盾的流式输出]
# =====================================================
@torch.no_grad()
def generate_response(model, tokenizer, prompt_text, persona_name):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)
    generated_ids = input_ids.clone()

    print(f"🤖 小乐 ({persona_name}形态): ", end="", flush=True)
    
    for _ in range(MAX_NEW_TOKENS):
        outputs = model(generated_ids)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        next_token_logits = logits[:, -1, :]
        
        # 🛡️ 启动重复惩罚护盾
        for token_id in set(generated_ids[0].tolist()):
            if next_token_logits[0, token_id] < 0:
                next_token_logits[0, token_id] *= REP_PENALTY
            else:
                next_token_logits[0, token_id] /= REP_PENALTY
                
        # 温度与核采样
        next_token_logits = next_token_logits / TEMPERATURE
        # 防止数值溢出或异常
        next_token_logits = torch.nan_to_num(next_token_logits, nan=-1e4, posinf=1e4, neginf=-1e4)
        
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > TOP_P
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        next_token_logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        if next_token.item() in [tokenizer.eos_token_id, 151645]:
            break
            
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        word = tokenizer.decode([next_token.item()])
        print(word, end="", flush=True)
        
    print() # 换行

# =====================================================
# [主程序：双脑区热切换终端]
# =====================================================
def main():
    print(f"🚀 初始化多脑区终极控制台 | 硬件: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    config = ModelConfig(
        vocab_size=151936, hidden_dim=1024, num_layers=24, 
        num_heads=16, num_kv_heads=16, initial_ffn_dim=2816
    )
    model = GrowableLLM(config).to(DEVICE)
    
    print(f"💉 正在探测大盘物理维度: {WEIGHT_PATH}...")
    ckpt = torch.load(WEIGHT_PATH, map_location=DEVICE, weights_only=True)
    
    # 🌟 极客设计：自动计算所有阶段累加的额外维度！
    target_ffn_dim = ckpt["blocks.0.ffn.gate_proj.weight"].shape[0]
    need_expand = target_ffn_dim - config.initial_ffn_dim
    if need_expand > 0:
        print(f"🧊 探测到前期累计拓展 {need_expand} 维，正在瞬间恢复骨架...")
        model.expand_model(extra_dim=need_expand)
        
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    
    # 默认唤醒“老中医”脑区
    current_persona = "med"
    persona_names = {"chat": "高情商", "med": "老中医"}
    
    # 🌟 初始化时，强行注入当前脑区的路由掩码
    apply_brain_routing(model, current_persona)

    print("\n" + "="*60)
    print("🌿 赛博老中医坐堂测试中... (双脑隔离系统已上线)")
    print("💡 快捷指令：")
    print("   输入 '/1' 或 '/chat' 切换至 [高情商闲聊模式] (保留至 3072 维，中医层物理断电)")
    print("   输入 '/2' 或 '/med'  切换至 [老中医问诊模式] (全功率 3200 维) - 当前默认")
    print("   输入 'q' 或 'exit'   退出系统")
    print("="*60 + "\n")

    while True:
        try:
            mode_tag = persona_names[current_persona]
            user_input = input(f"\n👤 [{mode_tag}] 你的问题: ")
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("👋 系统已休眠！")
                break
                
            # 🌟 捕获切换指令，瞬间切换底层物理路由器！
            elif user_input.lower() in ['/1', '/chat']:
                current_persona = "chat"
                apply_brain_routing(model, current_persona)
                print("🔄 已热切换至：高情商脑区 (中医层已物理断电)")
                continue
            elif user_input.lower() in ['/2', '/med']:
                current_persona = "med"
                apply_brain_routing(model, current_persona)
                print("🔄 已热切换至：老中医脑区 (全维度算力倾泻)")
                continue
            elif not user_input.strip():
                continue
                
            # 根据当前状态，动态提取密钥
            system_prompt = PROMPTS[current_persona]
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_input}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            generate_response(model, tokenizer, prompt, persona_names[current_persona])
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 强制终止，系统休眠！")
            break

if __name__ == "__main__":
    main()