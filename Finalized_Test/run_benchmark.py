import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset, load_from_disk
from models import GrowableLLM, ModelConfig

# Windows GBK 兼容：确保 stdout 支持 UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "buffer"):
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

# =============================================
# 从统一配置文件读取
# =============================================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
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
    config.torch_dtype = torch.bfloat16

    # 3. 在 CPU 创建 HF 模型，避免 GPU 双模型 OOM
    hf_model = AutoModelForCausalLM.from_config(config)
    hf_sd = hf_model.state_dict()

    # 4. 只取 custom model 的 weights，避免构建完整 state_dict（CPU 内存优化）
    custom_sd = custom_model.state_dict()

    hf_sd["model.embed_tokens.weight"] = custom_sd.pop("embed.weight").cpu()
    hf_sd["model.norm.weight"] = custom_sd.pop("norm.weight").cpu()
    hf_sd["lm_head.weight"] = custom_sd.pop("lm_head.weight").cpu()

    for i in range(config.num_hidden_layers):
        layer_keys = [
            f"blocks.{i}.attn_norm.weight",
            f"blocks.{i}.ffn_norm.weight",
            f"blocks.{i}.attn.q_proj.weight",
            f"blocks.{i}.attn.k_proj.weight",
            f"blocks.{i}.attn.v_proj.weight",
            f"blocks.{i}.attn.o_proj.weight",
            f"blocks.{i}.ffn.gate_proj.weight",
            f"blocks.{i}.ffn.up_proj.weight",
            f"blocks.{i}.ffn.down_proj.weight",
        ]
        targets = [
            f"model.layers.{i}.input_layernorm.weight",
            f"model.layers.{i}.post_attention_layernorm.weight",
            f"model.layers.{i}.self_attn.q_proj.weight",
            f"model.layers.{i}.self_attn.k_proj.weight",
            f"model.layers.{i}.self_attn.v_proj.weight",
            f"model.layers.{i}.self_attn.o_proj.weight",
            f"model.layers.{i}.mlp.gate_proj.weight",
            f"model.layers.{i}.mlp.up_proj.weight",
            f"model.layers.{i}.mlp.down_proj.weight",
        ]
        for src, tgt in zip(layer_keys, targets):
            hf_sd[tgt] = custom_sd.pop(src).cpu()

    hf_model.load_state_dict(hf_sd)
    # 清理中间大 dict 后移到 GPU
    del custom_sd, hf_sd
    hf_model = hf_model.to(DEVICE)
    print("✨ HF 桥接完成！模型现已无缝接入全宇宙开源生态。")
    return hf_model

# =====================================================
# 4. 评测引擎：EleutherAI / lm-evaluation-harness
# =====================================================
def run_lm_evaluation(model, tokenizer, model_name, limit=None):
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

    eval_kwargs = dict(model=lm_obj, tasks=tasks, num_fewshot=0)
    if limit is not None:
        eval_kwargs["limit"] = limit
        print(f"⚡ 调试模式: 仅评测前 {limit} 条样本")

    print("⏳ 开始疯狂跑分，请耐心等待 (进度将全自动输出)...")
    results = lm_eval.simple_evaluate(**eval_kwargs)

    # 打印超级震撼的工业级量化表格
    print("\n" + "="*60)
    print(f"🏆 [{model_name}] 终极能力量化表:")
    print(lm_eval.utils.make_table(results))
    print("="*60)

    # 自动保存为标准 JSON 分析报告（紧凑格式避免超大文件）
    report_file = f"eval_report_{model_name.lower()}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=None, ensure_ascii=False, default=str)
    print(f"💾 详细量化测试报告已保存至: {report_file}")

    # 清理 HF bridge 模型释放 GPU 显存
    del hf_model, lm_obj
    torch.cuda.empty_cache()


# =====================================================
# 5. 辅助工具
# =====================================================
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def save_ppl(ppl, model_name):
    path = f"eval_report_{model_name.lower()}_ppl.json"
    with open(path, "w") as f:
        json.dump({"model": model_name, "wikitext2_ppl": ppl}, f, indent=2)
    print(f"💾 PPL 结果已保存至: {path}")

def extract_gsm8k_metrics(results):
    try:
        gsm8k = results.get("results", {}).get("gsm8k", {})
        return {
            "exact_match": gsm8k.get("exact_match,strict-match", None),
            "exact_match_stderr": gsm8k.get("exact_match,strict-match_stderr", None),
        }
    except Exception:
        return {"error": "parse failed"}

def print_comparison(results_list):
    print("\n" + "="*70)
    print("📊 Baseline vs Master 对比结果")
    print("="*70)
    print(f"\n{'指标':<25} {'Baseline':<15} {'Master':<15} {'Δ':<15}")
    print("-"*70)

    baseline = next(r for r in results_list if r["model"] == "Baseline")
    master = next(r for r in results_list if r["model"] == "Master")

    if baseline.get("ppl") is not None and master.get("ppl") is not None:
        delta = master["ppl"] - baseline["ppl"]
        print(f"{'WikiText-2 PPL ↓':<25} {baseline['ppl']:<15.4f} {master['ppl']:<15.4f} {delta:<+15.4f}")

    b_acc = baseline.get("gsm8k_acc")
    m_acc = master.get("gsm8k_acc")
    if b_acc is not None and m_acc is not None:
        delta = m_acc - b_acc
        print(f"{'GSM8K Acc ↑':<25} {b_acc:<15.2%} {m_acc:<15.2%} {delta:<+15.2%}")

    print("="*70)

def run_eval_model(model_loader_fn, tokenizer, do_ppl, do_gsm8k, limit=None):
    model, model_name = model_loader_fn()
    result = {"model": model_name}

    if do_ppl:
        ppl = evaluate_wikitext_ppl(model, tokenizer)
        save_ppl(ppl, model_name)
        result["ppl"] = ppl

    if do_gsm8k:
        raw = run_lm_evaluation(model, tokenizer, model_name, limit=limit)
        metrics = extract_gsm8k_metrics(raw) if raw else None
        if metrics and "exact_match" in metrics and metrics["exact_match"] is not None:
            result["gsm8k_acc"] = float(metrics["exact_match"]) / 100
            print(f"✅ [{model_name}] GSM8K Acc = {result['gsm8k_acc']:.2%}")

    # 释放模型显存
    del model
    torch.cuda.empty_cache()
    return result

# =====================================================
# 6. 主控台 / CLI
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GrowableLLM 评测工具")
    parser.add_argument("--model", choices=["baseline", "master", "both"], help="评测模型")
    parser.add_argument("--task", choices=["ppl", "gsm8k", "both"], help="评测任务")
    parser.add_argument("--limit", type=int, default=None, help="GSM8K 限制样本数（调试用）")
    args = parser.parse_args()

    # ---------- 自动模式 ----------
    if args.model and args.task:
        tokenizer = get_tokenizer()
        do_ppl = args.task in ("ppl", "both")
        do_gsm8k = args.task in ("gsm8k", "both")

        if args.model == "both":
            all_results = []
            print("="*70)
            print("🧪 GrowableLLM 自动批量评测 (Baseline + Master)")
            print("="*70)

            all_results.append(run_eval_model(load_baseline, tokenizer, do_ppl, do_gsm8k, limit=args.limit))
            torch.cuda.empty_cache()
            all_results.append(run_eval_model(load_master, tokenizer, do_ppl, do_gsm8k, limit=args.limit))

            print_comparison(all_results)
            print("\n🎉 全部评测完成！")
        else:
            loader = load_baseline if args.model == "baseline" else load_master
            run_eval_model(loader, tokenizer, do_ppl, do_gsm8k, limit=args.limit)

    # ---------- 交互模式 ----------
    else:
        print("="*60)
        print("🧪 GrowableLLM 终极 A/B 量化评测台 (EleutherAI Edition)")
        print("="*60)

        tokenizer = get_tokenizer()

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
            ppl = evaluate_wikitext_ppl(model, tokenizer)
            save_ppl(ppl, model_name)

        if task_choice in ['2', '3']:
            run_lm_evaluation(model, tokenizer, model_name, limit=args.limit)