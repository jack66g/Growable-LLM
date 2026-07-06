"""
Qwen1.5-0.5B Benchmark 脚本
测试 WikiText-2 PPL 和 GSM8K
"""

import os
import sys

# 确保能导入根目录的 models_hf.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 设置 HF 镜像（必须在 import datasets 之前）
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from models_hf import GrowableQwen2ForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen1.5-0.5B"

# =====================================================
# 1. 加载模型
# =====================================================
print("=" * 50)
print("Loading Qwen model...")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载 Phase 2 权重
WEIGHTS_PATH = "/home/tbnl/Growable-LLM/Experiment_Replication/growable_qwen_med_expert_epoch3.pt"

model = GrowableQwen2ForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
# Phase 1: +256, Phase 2: +128, 总计 +384
model.expand_model(extra_dim=384)  # 2816 + 384 = 3200

try:
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights from: {WEIGHTS_PATH}")
except Exception as e:
    print(f"Failed to load weights: {e}")
    exit(1)

model.eval()
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# =====================================================
# 2. WikiText-2 PPL 测试
# =====================================================
print("\n" + "=" * 50)
print("WikiText-2 Perplexity Evaluation")
print("=" * 50)

wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# 取前 500 tokens 的段落
texts = [t for t in wikitext["text"] if len(t.strip()) > 100][:100]

total_loss = 0
total_tokens = 0

MAX_LENGTH = 512

with torch.no_grad():
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        input_ids = inputs["input_ids"].to(DEVICE)

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        total_loss += loss.item() * input_ids.shape[1]
        total_tokens += input_ids.shape[1]

        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{len(texts)} texts...")

ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
print(f"\nWikiText-2 PPL: {ppl:.4f}")

# =====================================================
# 3. GSM8K 测试 (少样本)
# =====================================================
print("\n" + "=" * 50)
print("GSM8K Evaluation (5-shot)")
print("=" * 50)

gsm8k = load_dataset("openai/gsm8k", "main", split="test")

# 取 5 个样本作为 few-shot examples
few_shot_examples = []
for i in range(5):
    few_shot_examples.append(f"Question: {gsm8k[i]['question']}\nAnswer: {gsm8k[i]['answer']}")

# 测试 20 个问题
test_questions = gsm8k[5:25]

def extract_answer(response):
    # 简单提取数字答案
    import re
    numbers = re.findall(r'\$?(\d+(?:\.\d+)?)\$?', response)
    if numbers:
        return numbers[-1]
    return None

correct = 0
total = len(test_questions)

with torch.no_grad():
    for i, q in enumerate(test_questions):
        prompt = "\n\n".join(few_shot_examples) + f"\n\nQuestion: {q['question']}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)

        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        # 提取答案
        pred = extract_answer(response)
        true_ans = extract_answer(q['answer'])

        if pred and true_ans and pred == true_ans:
            correct += 1

        if (i + 1) % 5 == 0:
            print(f"Processed {i+1}/{total} questions, Accuracy: {correct}/{i+1}")

accuracy = correct / total * 100
print(f"\nGSM8K Accuracy: {accuracy:.1f}% ({correct}/{total})")

# =====================================================
# 总结
# =====================================================
print("\n" + "=" * 50)
print("Benchmark Results")
print("=" * 50)
print(f"WikiText-2 PPL: {ppl:.4f}")
print(f"GSM8K Accuracy: {accuracy:.1f}%")
print("=" * 50)
