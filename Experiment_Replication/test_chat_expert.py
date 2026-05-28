import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import GrowableLLM, ModelConfig

# =====================================================
# [配置区]
# =====================================================
MODEL_PATH = "growable_chat_expert_epoch3.pth" # 刚刚出炉的高情商大盘
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.85              
REPETITION_PENALTY = 1.15  # 专治“复读机”

# =====================================================
# 1. 启动与模型自适应加载
# =====================================================
print(f"🚀 加载设备: {DEVICE}")
print("⚙️ 加载 Qwen Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 初始化千问 0.5B 的原始骨架
config = ModelConfig(
    vocab_size=151936, hidden_dim=1024, num_layers=24, 
    num_heads=16, num_kv_heads=16, initial_ffn_dim=2816
)
model = GrowableLLM(config).to(DEVICE)

print(f"💉 正在读取高情商权重: {MODEL_PATH}")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

# 自动探测权重里的 FFN 维度，自动扩容！
target_ffn_dim = ckpt["blocks.0.ffn.gate_proj.weight"].shape[0]
base_dim = config.initial_ffn_dim
need_expand = target_ffn_dim - base_dim

if need_expand > 0:
    print(f"🌱 探测到额外生长突触，自动物理扩容: +{need_expand} 维")
    model.expand_model(extra_dim=need_expand)

# 严格加载所有权重
model.load_state_dict(ckpt, strict=True)
model.eval()
print("✅ 模型灵魂注入完成！\n")

# =====================================================
# 2. 推理函数 (流式思维生成)
# =====================================================
@torch.inference_mode()
def chat(user_input):
    # 🌟 必须与第一阶段特训时的 System Prompt 一字不差！
    prompt = (
        f"<|im_start|>system\n你是一个高情商、乐于助人的AI助手。<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    generated = input_ids.clone()
    generated_list = []

    for _ in range(MAX_NEW_TOKENS):
        logits, _ = model(generated)
        next_token_logits = logits[:, -1, :].clone()

        # 1. 应用重复惩罚
        if len(generated_list) > 0:
            for token_id in set(generated_list):
                if next_token_logits[0, token_id] < 0:
                    next_token_logits[0, token_id] *= REPETITION_PENALTY
                else:
                    next_token_logits[0, token_id] /= REPETITION_PENALTY

        # 2. 温度平滑
        next_token_logits = next_token_logits / TEMPERATURE
        next_token_logits = torch.nan_to_num(next_token_logits, nan=-1e4, posinf=1e4, neginf=-1e4)

        # 3. Top-P 核采样
        probs = F.softmax(next_token_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > TOP_P
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0 
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True) 

        # 4. 随机采样
        next_token = torch.multinomial(probs, num_samples=1)
        generated_list.append(next_token.item())
        generated = torch.cat([generated, next_token], dim=1)

        # 遇到 <|im_end|> 或 EOS 停止
        if next_token.item() == tokenizer.eos_token_id or next_token.item() == 151645: # 151645 是 Qwen 的 <|im_end|>
            break

    # 解码
    output = tokenizer.decode(generated[0], skip_special_tokens=False)

    # 提取回答部分
    if "<|im_start|>assistant\n" in output:
        output = output.split("<|im_start|>assistant\n")[-1]
    if "<|im_end|>" in output:
        output = output.split("<|im_end|>")[0]

    return output.strip()

# =====================================================
# 3. CLI 交互终端
# =====================================================
print("=" * 60)
print("😊 高情商聊天大盘启动成功！")
print("指令: 输入文字对话 | 输入 exit 退出")
print("=" * 60)

while True:
    query = input("\n🧑 你: ")
    if query.lower() in ["exit", "quit"]:
        break
    if not query.strip():
        continue

    try:
        response = chat(query)
        print(f"\n🤖 小乐: {response}")
    except Exception as e:
        print(f"\n❌ 推理异常: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()