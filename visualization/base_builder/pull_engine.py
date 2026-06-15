import os
import sys
import time
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models import GrowableLLM, ModelConfig

try:
    # 🌟 新增导入 AutoTokenizer
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("⚠️ 缺少 transformers 库，请先运行: pip install transformers")
    sys.exit(1)

class Color:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(title):
    print(f"\n{Color.CYAN}={'='*58}={Color.RESET}")
    print(f"{Color.BOLD} {title}{Color.RESET}")
    print(f"{Color.CYAN}={'='*58}={Color.RESET}\n")

def run_pull_engine():
    """
    核心功能：拉取/扫描基座模型，物理重塑，固化 Tokenizer，并向主程序返回 (传参) 模型配置。
    Returns:
        dict: 包含模型路径、Tokenizer路径和骨骼参数的字典，供训练引擎使用。
    """
    print_header("📥 基座引擎构建与参数提取 (Base Model Puller)")
    
    online_dir = "./online_base_models"
    local_dir = "./local_upload_models"
    trained_dir = "./trained_weights" # 🌟 新增：存放已拓展变异模型的目录
    os.makedirs(online_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(trained_dir, exist_ok=True)

    print(f"{Color.BOLD}【第一步】选择基座模型操作：{Color.RESET}")
    print("  [1] 🌐 线上扒取 (下载并转换 -> 保存至 online_base_models -> 传参)")
    print("  [2] 📂 扫描本地 (读取用户上传目录 -> 转换 -> 传参)")
    print(f"  [3] 🚀 {Color.GREEN}选择已扒取底座{Color.RESET} (直接加载已存在的模型参数 -> 传参)")
    print(f"  [4] 🧬 {Color.YELLOW}选择已拓展变异模型{Color.RESET} (加载 trained_weights 中的模型 -> 继续套娃训练)")
    source_choice = input(f"> 请选择 (1/2/3/4): ").strip()

    model_path_or_id = ""
    save_target_dir = ""

    # =========================================================
    # 🌟 新增选项 4: 直接读取训练过的变异模型配置并传参
    # =========================================================
    if source_choice == '4':
        print(f"\n{Color.BOLD}【第二步】选择已拓展的变异模型 ({trained_dir}){Color.RESET}")
        pt_files = [f for f in os.listdir(trained_dir) if f.endswith(".pt")]
        
        if not pt_files:
            print(f"{Color.RED}❌ 目录为空！请先进行一次训练生成变异模型。{Color.RESET}")
            return None
            
        for i, f_name in enumerate(pt_files):
            print(f"  [{i+1}] {f_name}")
            
        m_choice = input(f"> 请选择要继续拓展的模型序号: ").strip()
        if m_choice.isdigit() and 1 <= int(m_choice) <= len(pt_files):
            selected_file = pt_files[int(m_choice)-1]
        else:
            selected_file = pt_files[0]
            
        pt_path = os.path.join(trained_dir, selected_file)
        config_path = pt_path.replace(".pt", "_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                
            print(f"\n{Color.GREEN}🔍 【变异基因读取完毕】瞬间读取到拓展后的物理骨骼数据:{Color.RESET}")
            print(f"   - 词表大小 (Vocab Size) : {Color.YELLOW}{config_dict['vocab_size']}{Color.RESET}")
            print(f"   - 隐藏层维 (Hidden Dim) : {Color.YELLOW}{config_dict['hidden_dim']}{Color.RESET}")
            print(f"   - 网络深度 (Layers)     : {Color.YELLOW}{config_dict['num_layers']}{Color.RESET}")
            print(f"   - 注意力头 (Attn Heads) : {Color.YELLOW}{config_dict['num_heads']}{Color.RESET}")
            print(f"   - KV缓存头 (KV Heads)   : {Color.YELLOW}{config_dict['num_kv_heads']}{Color.RESET}")
            print(f"   - FFN宽幅 (FFN Dim)     : {Color.YELLOW}{config_dict['initial_ffn_dim']} (含已拓展维度){Color.RESET}")
            print(f"   - 密码本 (Tokenizer)    : {Color.YELLOW}{config_dict.get('tokenizer_path', '未记录')}{Color.RESET}")
            print("-" * 50)
            print(f"\n{Color.CYAN}[System] 变异参数已获取！即将向主控制台传参...{Color.RESET}")
            time.sleep(1)
            return config_dict 
        else:
            print(f"{Color.RED}❌ 找不到伴生的 _config.json 参数文件，该模型无法进行套娃拓展。{Color.RESET}")
            return None

    # =========================================================
    # 选项 3: 直接读取已扒取的模型配置并传参
    # =========================================================
    elif source_choice == '3':
        print(f"\n{Color.BOLD}【第二步】选择已准备好的线上底座 ({online_dir}){Color.RESET}")
        pt_files = [f for f in os.listdir(online_dir) if f.endswith(".pt")]
        
        if not pt_files:
            print(f"{Color.RED}❌ 目录为空！请先使用 [1] 线上扒取模型。{Color.RESET}")
            return None
            
        for i, f_name in enumerate(pt_files):
            print(f"  [{i+1}] {f_name}")
            
        m_choice = input(f"> 请选择要被训练的模型序号: ").strip()
        if m_choice.isdigit() and 1 <= int(m_choice) <= len(pt_files):
            selected_file = pt_files[int(m_choice)-1]
        else:
            selected_file = pt_files[0]
            
        pt_path = os.path.join(online_dir, selected_file)
        config_path = pt_path.replace(".pt", "_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                
            print(f"\n{Color.GREEN}🔍 【雷达扫描完毕】瞬间读取到底座的物理骨骼数据:{Color.RESET}")
            print(f"   - 词表大小 (Vocab Size) : {Color.YELLOW}{config_dict['vocab_size']}{Color.RESET}")
            print(f"   - 隐藏层维 (Hidden Dim) : {Color.YELLOW}{config_dict['hidden_dim']}{Color.RESET}")
            print(f"   - 网络深度 (Layers)     : {Color.YELLOW}{config_dict['num_layers']}{Color.RESET}")
            print(f"   - 注意力头 (Attn Heads) : {Color.YELLOW}{config_dict['num_heads']}{Color.RESET}")
            print(f"   - KV缓存头 (KV Heads)   : {Color.YELLOW}{config_dict['num_kv_heads']}{Color.RESET}")
            print(f"   - FFN宽幅 (FFN Dim)     : {Color.YELLOW}{config_dict['initial_ffn_dim']}{Color.RESET}")
            print(f"   - 密码本 (Tokenizer)    : {Color.YELLOW}{config_dict.get('tokenizer_path', '未记录')}{Color.RESET}")
            print("-" * 50)
            print(f"\n{Color.CYAN}[System] 参数已获取！即将向主控制台传参...{Color.RESET}")
            time.sleep(1)
            return config_dict 
        else:
            print(f"{Color.RED}❌ 找不到伴生的 _config.json 参数文件，请重新执行 [1] 扒取该模型。{Color.RESET}")
            return None

    # =========================================================
    # 选项 2: 扫描本地并转换
    # =========================================================
    elif source_choice == '2':
        print(f"\n{Color.BOLD}【第二步】扫描本地文件夹 ({local_dir}){Color.RESET}")
        local_models = [d for d in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, d))]
        
        if not local_models:
            print(f"{Color.RED}❌ 本地文件夹为空！请先将模型丢进 {local_dir} 文件夹中。{Color.RESET}")
            return None
            
        for i, m_name in enumerate(local_models):
            print(f"  [{i+1}] {m_name}")
            
        m_choice = input(f"> 请选择要作为底座的模型序号: ").strip()
        if m_choice.isdigit() and 1 <= int(m_choice) <= len(local_models):
            selected_model = local_models[int(m_choice)-1]
        else:
            selected_model = local_models[0]
            
        model_path_or_id = os.path.join(local_dir, selected_model)
        save_target_dir = local_dir
        print(f"[*] 已锁定本地模型: {Color.GREEN}{model_path_or_id}{Color.RESET}")

    # =========================================================
    # 选项 1: 线上扒取模式
    # =========================================================
    elif source_choice == '1':
        print(f"\n{Color.BOLD}【第二步】网络通道与模型地址配置{Color.RESET}")
        
        use_mirror = input(f"🚀 是否走国内加速通道下载 (hf-mirror.com)? (Y/n，直接回车默认 Y): ").strip().upper()
        if use_mirror != 'N':
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print(f"{Color.GREEN}[*] 国内加速节点已挂载！{Color.RESET}")
        else:
            print(f"[*] 使用官方默认节点。")

        print(f"\n📝 请输入要扒取的 HuggingFace 模型地址")
        model_id = input(f"{Color.YELLOW}> 直接回车默认走 [Qwen/Qwen2.5-0.5B]: {Color.RESET}").strip()
        
        if not model_id:
            model_id = "Qwen/Qwen2.5-0.5B"
            
        model_path_or_id = model_id
        save_target_dir = online_dir
        print(f"[*] 目标已锁定: {Color.GREEN}{model_path_or_id}{Color.RESET}")
        
    else:
        print(f"{Color.RED}❌ 无效选项。{Color.RESET}")
        return None

    # =========================================================
    # 第三步：雷达扫描与真正的物理重塑 (选项1和2共用逻辑)
    # =========================================================
    print(f"\n{Color.BOLD}【第三步】底层结构解析与物理重塑{Color.RESET}")
    print(f"⏳ 正在启动真实探测引擎解析 [{model_path_or_id}] ...")
    
    try:
        hf_config = AutoConfig.from_pretrained(model_path_or_id)
        
        v_size = getattr(hf_config, "vocab_size", 151936)
        h_dim = getattr(hf_config, "hidden_size", getattr(hf_config, "n_embd", 1024))
        layers = getattr(hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", 24))
        attn_h = getattr(hf_config, "num_attention_heads", getattr(hf_config, "n_head", 16))
        kv_h = getattr(hf_config, "num_key_value_heads", attn_h)
        ffn_dim = getattr(hf_config, "intermediate_size", 2816)

        approx_params = (v_size * h_dim) + (layers * (h_dim * h_dim * 4 + h_dim * ffn_dim * 3))
        scale_str = f"{approx_params / 1e9:.1f}B" if approx_params > 1e9 else f"{approx_params / 1e6:.0f}M"

        print(f"\n{Color.GREEN}🔍 【雷达扫描完毕】探测到 {scale_str} 级模型的真实骨骼数据:{Color.RESET}")
        print(f"   - 词表大小 (Vocab Size) : {Color.YELLOW}{v_size}{Color.RESET}")
        print(f"   - 隐藏层维 (Hidden Dim) : {Color.YELLOW}{h_dim}{Color.RESET}")
        print(f"   - 网络深度 (Layers)     : {Color.YELLOW}{layers}{Color.RESET}")
        print(f"   - 注意力头 (Attn Heads) : {Color.YELLOW}{attn_h}{Color.RESET}")
        print(f"   - KV缓存头 (KV Heads)   : {Color.YELLOW}{kv_h}{Color.RESET}")
        print(f"   - FFN宽幅 (FFN Dim)     : {Color.YELLOW}{ffn_dim}{Color.RESET}")
        print("-" * 50)
        
        # 🌟 核心新增：拉取并固化 Tokenizer
        print(f"\n{Color.CYAN}⏳ 正在拉取并固化 Tokenizer (母语密码本)...{Color.RESET}")
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, trust_remote_code=True)
        safe_model_name = model_path_or_id.split('/')[-1]
        tokenizer_save_dir = os.path.join(save_target_dir, safe_model_name + "_tokenizer")
        tokenizer.save_pretrained(tokenizer_save_dir)
        print(f"{Color.GREEN}✅ Tokenizer 已物理固化至: {tokenizer_save_dir}{Color.RESET}")

        print(f"\n{Color.CYAN}⏳ 正在拉取原始权重洪流 (受网速影响可能需要几分钟，请耐心等待)...{Color.RESET}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id, 
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        hf_state_dict = hf_model.state_dict()
        
        print(f"{Color.CYAN}🚀 正在初始化你手写的 GrowableLLM 物理沙盒...{Color.RESET}")
        config = ModelConfig(
            vocab_size=v_size,
            hidden_dim=h_dim,
            num_layers=layers,
            num_heads=attn_h,
            num_kv_heads=kv_h,
            initial_ffn_dim=ffn_dim,
            rope_theta=getattr(hf_config, "rope_theta", 1000000), 
        )
        custom_model = GrowableLLM(config)
        custom_state_dict = custom_model.state_dict()
        
        print(f"{Color.CYAN}🔄 开始高维度映射权重矩阵 (外科手术式移植)...{Color.RESET}")
        mapping_count = 0
        
        for name, param in hf_state_dict.items():
            if name == "model.embed_tokens.weight":
                custom_state_dict["embed.weight"].copy_(param)
                mapping_count += 1
            elif name == "model.norm.weight":
                custom_state_dict["norm.weight"].copy_(param)
                mapping_count += 1
            elif name == "lm_head.weight":
                custom_state_dict["lm_head.weight"].copy_(param)
                mapping_count += 1
                
            elif name.startswith("model.layers."):
                parts = name.split(".")
                layer_idx = parts[2]
                
                # 归一化层
                if "input_layernorm.weight" in name:
                    custom_state_dict[f"blocks.{layer_idx}.attn_norm.weight"].copy_(param)
                    mapping_count += 1
                elif "post_attention_layernorm.weight" in name:
                    custom_state_dict[f"blocks.{layer_idx}.ffn_norm.weight"].copy_(param)
                    mapping_count += 1
                    
                # Attention 层
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
                    
                # FFN (专家层)
                elif "mlp.gate_proj.weight" in name:
                    custom_state_dict[f"blocks.{layer_idx}.ffn.gate_proj.weight"].copy_(param)
                    mapping_count += 1
                elif "mlp.up_proj.weight" in name:
                    custom_state_dict[f"blocks.{layer_idx}.ffn.up_proj.weight"].copy_(param)
                    mapping_count += 1
                elif "mlp.down_proj.weight" in name:
                    custom_state_dict[f"blocks.{layer_idx}.ffn.down_proj.weight"].copy_(param)
                    mapping_count += 1

        if "lm_head.weight" not in hf_state_dict:
            custom_state_dict["lm_head.weight"].copy_(custom_state_dict["embed.weight"])
            mapping_count += 1

        custom_model.load_state_dict(custom_state_dict)
        print(f"{Color.GREEN}✅ 成功映射 {mapping_count} 个张量！{Color.RESET}")
        
        # 保存模型权重
        final_save_path = os.path.join(save_target_dir, safe_model_name + "_growable.pt")
        torch.save(custom_model.state_dict(), final_save_path)
        
        # 🌟 核心：保存骨骼参数配置到 JSON (包含 tokenizer_path)
        config_dict = {
            "model_path": final_save_path,
            "tokenizer_path": tokenizer_save_dir,
            "vocab_size": v_size,
            "hidden_dim": h_dim,
            "num_layers": layers,
            "num_heads": attn_h,
            "num_kv_heads": kv_h,
            "initial_ffn_dim": ffn_dim
        }
        
        config_save_path = final_save_path.replace(".pt", "_config.json")
        with open(config_save_path, "w", encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4)
            
        print(f"\n🎉 完美收工！专属底座及参数文件已保存至: {Color.GREEN}{save_target_dir}{Color.RESET}")
        print(f"{Color.CYAN}[System] 参数已就绪，准备向主控制台传参...{Color.RESET}")
        time.sleep(1)
        
        return config_dict 
        
    except Exception as e:
        print(f"\n{Color.RED}❌ 任务失败，原因: {str(e)}{Color.RESET}")
        return None

if __name__ == "__main__":
    # 纯净的直接执行，不再使用假数据
    result = run_pull_engine()
    if result:
        print(f"\n[返回的传参 Payload]: \n{json.dumps(result, indent=2, ensure_ascii=False)}")