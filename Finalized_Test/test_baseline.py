import os
import json
import torch
from transformers import AutoTokenizer
from models import GrowableLLM, ModelConfig

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    full_config = json.load(f)

model_cfg = full_config["model"]
tk_cfg = full_config["tokenizer"]
paths = full_config["paths"]
inf_cfg = full_config["inference"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("==================================================")
print("⚖️ 初始化 [Baseline] SmolLM2-360M-Instruct (原始基座) ...")
print("==================================================")

# 1. 初始化 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(tk_cfg["model_id"])

# 2. 载入原始 Config (不进行任何正交扩容)
config = ModelConfig(**model_cfg)
model = GrowableLLM(config).to(DEVICE)

# 3. 直接加载原始基座权重
print("⏳ 正在注入原始基座权重...")
try:
    model.load_state_dict(torch.load(paths["base_weights"], weights_only=True))
    model.eval()
    print("✅ 原始基座已苏醒！准备接受 A/B 测试。\n")
except Exception as e:
    print(f"❌ 权重加载失败，错误: {e}")
    exit()

# ==========================================
# 推理生成引擎
# ==========================================
def generate_code(prompt, max_new_tokens=None, temperature=None):
    if max_new_tokens is None:
        max_new_tokens = inf_cfg["max_new_tokens"]
    if temperature is None:
        temperature = inf_cfg["temperature"]

    system_prompt = "Think step by step mathematically before writing code. Be concise."

    chat_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    input_ids = tokenizer.encode(chat_prompt, return_tensors="pt").to(DEVICE)
    prompt_length = input_ids.shape[1]

    print("-" * 50)
    print("🤖 [Baseline]:\n")

    generated_tokens = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out

            next_token_logits = logits[0, -1, :].clone()

            penalty_factor = inf_cfg["repetition_penalty"]
            for token_id in set(generated_tokens):
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= penalty_factor
                else:
                    next_token_logits[token_id] *= penalty_factor

            next_token_logits = next_token_logits / temperature

            top_k = inf_cfg["top_k"]
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('Inf')

            top_p = inf_cfg["top_p"]
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('Inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)

            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() in [tokenizer.eos_token_id, tokenizer.pad_token_id, 2]:
                break

    generated_ids = input_ids[0][prompt_length:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(output_text.strip())
    print("\n" + "=" * 50)

# ==========================================
# 🗣️ 交互式对话终端
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🟢 Baseline (原始) 终端已就绪！")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\n🧑‍💻 你: ")

            if user_input.strip().lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue

            generate_code(user_input)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
