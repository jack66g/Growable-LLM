import os
import json
import torch
from transformers import AutoTokenizer
from models import GrowableLLM, ModelConfig

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    full_config = json.load(f)

model_cfg = full_config["model"]
train_cfg = full_config["training"]
tk_cfg = full_config["tokenizer"]
paths = full_config["paths"]
inf_cfg = full_config["inference"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPAND_DIM = train_cfg["expand_dim"]

print("==================================================")
print("🚀 初始化 GrowableLLM [V3 Master Edition] ...")
print("==================================================")

# 1. 初始化 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(tk_cfg["model_id"])

# 2. 载入基座 Config
config = ModelConfig(**model_cfg)
model = GrowableLLM(config).to(DEVICE)

# 3. 还原正交扩张维度
print(f"📐 正在还原模型的正交扩张维度 (+{EXPAND_DIM}, +{EXPAND_DIM})...")
model.expand_model(extra_dim=EXPAND_DIM)
model.expand_model(extra_dim=EXPAND_DIM)

# 4. 加载终极权重
print("⏳ 正在注入 Master 灵魂...")
model.load_state_dict(torch.load(paths["master_weights"], weights_only=True))
model.eval()
print("✅ 模型已完全苏醒！\n")

# ==========================================
# 推理生成引擎
# ==========================================
def generate_code(prompt, max_new_tokens=None, temperature=None):
    if max_new_tokens is None:
        max_new_tokens = inf_cfg["max_new_tokens"]
    if temperature is None:
        temperature = inf_cfg["temperature"]

    chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(chat_prompt, return_tensors="pt").to(DEVICE)
    prompt_length = input_ids.shape[1]

    print("-" * 50)
    print("🤖 [GrowableLLM]:")

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out

            next_token_logits = logits[0, -1, :] / temperature
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.pad_token_id:
                break

    generated_ids = input_ids[0][prompt_length:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(output_text.strip())
    print("=" * 50)

# ==========================================
# 🗣️ 交互式对话终端
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🟢 GrowableLLM 终端已就绪！(输入 'exit' 或 'quit' 退出)")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\n🧑‍💻 你: ")

            if user_input.strip().lower() in ['exit', 'quit']:
                print("👋 拜拜！断开脑机接口。")
                break

            if not user_input.strip():
                continue

            generate_code(user_input)

        except KeyboardInterrupt:
            print("\n👋 强制中断，断开脑机接口。")
            break
        except Exception as e:
            print(f"\n❌ 发生严重错误: {e}")
