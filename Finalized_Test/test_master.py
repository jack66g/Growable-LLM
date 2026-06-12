import torch
from transformers import AutoTokenizer
from models import GrowableLLM, ModelConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("==================================================")
print("🚀 初始化 GrowableLLM [V3 Master Edition] ...")
print("==================================================")

# 1. 初始化 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")

# 2. 载入基座 Config
config = ModelConfig(
    vocab_size=49152, hidden_dim=960, num_layers=32, 
    num_heads=15, num_kv_heads=5, initial_ffn_dim=2560, rope_theta=100000
)
model = GrowableLLM(config).to(DEVICE)

# 3. 🛡️ 核心还原步骤：连续扩容两次，还原最终的物理形态
print("📐 正在还原模型的正交扩张维度 (Phase 1 & Phase 2)...")
model.expand_model(extra_dim=256) 
model.expand_model(extra_dim=256) 

# 4. 加载我们炼制完成的终极权重
print("⏳ 正在注入 Master 灵魂...")
model.load_state_dict(torch.load("GrowableLLM_360M_LogicCode_Master.pt", weights_only=True))
model.eval()
print("✅ 模型已完全苏醒！\n")

# ==========================================
# 推理生成引擎
# ==========================================
def generate_code(prompt, max_new_tokens=512, temperature=0.2):
    # 组装 ChatML 格式
    chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(chat_prompt, return_tensors="pt").to(DEVICE)
    prompt_length = input_ids.shape[1]
    
    print("-" * 50)
    print("🤖 [GrowableLLM]:")
    
    # 动态流式输出的效果（这里为了简便先一次性生成再打印，如果想更酷炫可以改流式）
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
            
            # 取出最后一个 Token 的 Logits 并应用 Temperature
            next_token_logits = logits[0, -1, :] / temperature
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 遇到结束符停止
            if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.pad_token_id:
                break
                
    generated_ids = input_ids[0][prompt_length:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(output_text.strip())
    print("=" * 50)

# ==========================================
# 🗣️ 交互式对话终端 (Interactive REPL)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🟢 GrowableLLM 终端已就绪！(输入 'exit' 或 'quit' 退出)")
    print("="*50 + "\n")
    
    while True:
        try:
            # 获取你的手动输入
            user_input = input("\n🧑‍💻 你: ")
            
            # 退出指令判断
            if user_input.strip().lower() in ['exit', 'quit']:
                print("👋 拜拜！断开脑机接口。")
                break
                
            # 防止空敲回车
            if not user_input.strip():
                continue
                
            # 扔给模型生成
            generate_code(user_input)
            
        except KeyboardInterrupt:
            # 捕获 Ctrl+C，优雅退出而不是疯狂报错
            print("\n👋 强制中断，断开脑机接口。")
            break
        except Exception as e:
            print(f"\n❌ 发生严重错误: {e}")