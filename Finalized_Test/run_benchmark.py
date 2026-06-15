import os
import json
import torch
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from models import GrowableLLM, ModelConfig

# =============================================
# 从统一配置文件读取
# =============================================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    full_config = json.load(f)

model_cfg = full_config["model"]
train_cfg = full_config["training"]
tk_cfg = full_config["tokenizer"]
paths = full_config["paths"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER_ID = tk_cfg["model_id"]
EXPAND_DIM = train_cfg["expand_dim"]

# =====================================================
# 1. 模型加载器
# =====================================================
def get_base_config():
    return ModelConfig(**model_cfg)

def load_baseline():
    print("\n⚖️ 正在加载 [Baseline] 原始基座模型...")
    config = get_base_config()
    model = GrowableLLM(config).to(DEVICE)
    try:
        model.load_state_dict(torch.load(paths["base_weights"], weights_only=True, map_location=DEVICE))
        model.eval()
        print("✅ Baseline 加载成功！")
        return model, "Baseline"
    except Exception as e:
        print(f"❌ Baseline 加载失败: {e}")
        exit()

def load_master():
    print("\n🚀 正在加载 [Master] GrowableLLM 特训模型...")
    config = get_base_config()
    model = GrowableLLM(config).to(DEVICE)

    print(f"📐 还原 Master 正交扩张维度 (+{EXPAND_DIM}, +{EXPAND_DIM})...")
    model.expand_model(extra_dim=EXPAND_DIM)
    model.expand_model(extra_dim=EXPAND_DIM)

    try:
        model.load_state_dict(torch.load(paths["master_weights"], weights_only=True, map_location=DEVICE))
        model.eval()
        print("✅ Master 加载成功！")
        return model, "Master"
    except Exception as e:
        print(f"❌ Master 加载失败: {e}")
        exit()

# =====================================================
# 2. 传统评测引擎：WikiText-2 PPL (测遗忘)
# =====================================================
@torch.no_grad()
def evaluate_wikitext_ppl(model, tokenizer, stride=512, max_length=1024):
    print("\n" + "-"*50)
    print("📚 开始测试 WikiText-2 PPL (常识与防遗忘测试)...")
    
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Evaluating PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        if input_ids.size(1) < 2:
            break

        logits, _ = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        nlls.append(loss * trg_len)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    print(f"🎯 最终 PPL 分数: {ppl.item():.4f}")
    return ppl.item()

# =====================================================
# 3. 🚀 核心黑科技：动态反向封装 HF 架构
# =====================================================
def convert_to_hf_format(custom_model):
    print("\n🔄 正在触发 [HF Bridge]：将私有架构逆向映射回 HuggingFace 标准形态...")
    
    # 1. 动态捕获当前模型的物理维度 (完美兼容你的动态扩容)
    ffn_dim = custom_model.blocks[0].ffn.current_dim
    
    # 2. 加载基座配置，并实时修改 FFN 维度
    config = AutoConfig.from_pretrained(TOKENIZER_ID)
    config.intermediate_size = ffn_dim
    
    # 3. 在内存中实例化一个空的官方标准模型
    hf_model = AutoModelForCausalLM.from_config(config).to(DEVICE)
    
    hf_sd = hf_model.state_dict()
    custom_sd = custom_model.state_dict()
    
    # 4. 执行逆向映射手术
    hf_sd["model.embed_tokens.weight"] = custom_sd["embed.weight"]
    hf_sd["model.norm.weight"] = custom_sd["norm.weight"]
    hf_sd["lm_head.weight"] = custom_sd["lm_head.weight"]
    
    for i in range(config.num_hidden_layers):
        hf_sd[f"model.layers.{i}.input_layernorm.weight"] = custom_sd[f"blocks.{i}.attn_norm.weight"]
        hf_sd[f"model.layers.{i}.post_attention_layernorm.weight"] = custom_sd[f"blocks.{i}.ffn_norm.weight"]
        
        hf_sd[f"model.layers.{i}.self_attn.q_proj.weight"] = custom_sd[f"blocks.{i}.attn.q_proj.weight"]
        hf_sd[f"model.layers.{i}.self_attn.k_proj.weight"] = custom_sd[f"blocks.{i}.attn.k_proj.weight"]
        hf_sd[f"model.layers.{i}.self_attn.v_proj.weight"] = custom_sd[f"blocks.{i}.attn.v_proj.weight"]
        hf_sd[f"model.layers.{i}.self_attn.o_proj.weight"] = custom_sd[f"blocks.{i}.attn.o_proj.weight"]
        
        hf_sd[f"model.layers.{i}.mlp.gate_proj.weight"] = custom_sd[f"blocks.{i}.ffn.gate_proj.weight"]
        hf_sd[f"model.layers.{i}.mlp.up_proj.weight"] = custom_sd[f"blocks.{i}.ffn.up_proj.weight"]
        hf_sd[f"model.layers.{i}.mlp.down_proj.weight"] = custom_sd[f"blocks.{i}.ffn.down_proj.weight"]
        
    hf_model.load_state_dict(hf_sd)
    print("✨ HF 桥接完成！模型现已无缝接入全宇宙开源生态。")
    return hf_model

# =====================================================
# 4. 评测引擎：EleutherAI / lm-evaluation-harness
# =====================================================
def run_lm_evaluation(model, tokenizer, model_name):
    print("\n" + "-"*50)
    print(f"🚀 正在启动全能评测平台 (lm-eval) 对 {model_name} 进行量化测试...")
    
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("❌ 错误: 未安装 lm-eval！请在新终端运行: pip install lm-eval")
        return

    # 开启代码运行许可环境变量 (EleutherAI 安全规范)
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # 调用桥接技术，将纯 PyTorch 模型伪装成 HF 模型
    hf_model = convert_to_hf_format(model)

    print("🔌 正在连接 lm-eval 引擎...")
    # 实例化 HFLM 对象，直接传入内存中的模型
    lm_obj = HFLM(pretrained=hf_model, tokenizer=tokenizer, batch_size="auto")

    # 选用现代测试集：
    # mbpp: Google 搞的经典 Python 代码测试集
    # gsm8k: 测数学逻辑 (逻辑能力是写好代码的核心底层能力)
    tasks = ["gsm8k"]
    print(f"📚 选定测试集: {tasks}")

    print("⏳ 开始疯狂跑分，请耐心等待 (进度将全自动输出)...")
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        num_fewshot=0,
        # limit=50,  # ⚠️ 调试开关：如果你只想跑 50 道题快速看结果，取消这行的注释。测全量就保持注释。
    )

    # 打印超级震撼的工业级量化表格
    print("\n" + "="*60)
    print(f"🏆 [{model_name}] 终极能力量化表:")
    print(lm_eval.utils.make_table(results))
    print("="*60)

    # 自动保存为标准 JSON 分析报告
    report_file = f"eval_report_{model_name.lower()}.json"
    with open(report_file, "w") as f:
        f.write(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"💾 详细量化测试报告已保存至: {report_file}")


# =====================================================
# 5. 主控台
# =====================================================
if __name__ == "__main__":
    print("="*60)
    print("🧪 GrowableLLM 终极 A/B 量化评测台 (EleutherAI Edition)")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    choice = input("\n请选择要评测的模型 (1: Baseline, 2: Master): ").strip()
    
    if choice == '1':
        model, model_name = load_baseline()
    elif choice == '2':
        model, model_name = load_master()
    else:
        print("无效输入，退出。")
        exit()
        
    print("\n请选择评测项目:")
    print("1. WikiText-2 PPL (测模型常识与遗忘)")
    print("2. LM-Eval 全自动测评 (主测代码 MBPP 与逻辑 GSM8K)")
    print("3. 两者都跑")
    task_choice = input("你的选择 (1/2/3): ").strip()
    
    if task_choice in ['1', '3']:
        evaluate_wikitext_ppl(model, tokenizer)
        
    if task_choice in ['2', '3']:
        run_lm_evaluation(model, tokenizer, model_name)