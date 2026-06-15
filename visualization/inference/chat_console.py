import os
import sys
import json
import time
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models import GrowableLLM, ModelConfig

class Color:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    print(f"\n{Color.CYAN}={'='*58}={Color.RESET}")
    print(f"{Color.BOLD} {title}{Color.RESET}")
    print(f"{Color.CYAN}={'='*58}={Color.RESET}\n")

def run_chat_console():
    clear_screen()
    print_header("💬 GrowableLLM 沉浸式交互控制台 (Inference UI)")
    
    # =========================================================
    # 1. 扫描并选择底座
    # =========================================================
    search_dirs = ["./trained_weights", "./online_base_models"]
    available_models = []
    
    print(f"{Color.BOLD}【第一步】雷达扫描可用模型...{Color.RESET}")
    for d in search_dirs:
        # 如果不是绝对路径，将相对于根目录拼接
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), d.strip("./"))
        if os.path.exists(base_dir):
            for f in os.listdir(base_dir):
                if f.endswith(".pt"):
                    pt_path = os.path.join(base_dir, f)
                    config_path = pt_path.replace(".pt", "_config.json")
                    if os.path.exists(config_path):
                        available_models.append((pt_path, config_path))

    if not available_models:
        print(f"{Color.RED}❌ 扫描不到任何带有 config 的模型！请先进行基座提取或训练。{Color.RESET}")
        return

    for i, (pt_p, cfg_p) in enumerate(available_models):
        # 只显示文件名，让界面更清爽
        print(f"  [{i+1}] {os.path.basename(pt_p)}")

    m_choice = input(f"\n> 请选择要唤醒的模型序号: ").strip()
    if m_choice.isdigit() and 1 <= int(m_choice) <= len(available_models):
        selected_pt, selected_cfg = available_models[int(m_choice)-1]
    else:
        selected_pt, selected_cfg = available_models[0]

    # =========================================================
    # 2. 加载基因与重塑沙盒
    # =========================================================
    print(f"\n{Color.BOLD}【第二步】物理沙盒唤醒中...{Color.RESET}")
    
    with open(selected_cfg, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{Color.GREEN}System{Color.RESET}] GPU 计算节点就绪: {device}")
    
    # 利用 config.json 中的数据动态构建大小匹配的躯壳
    model_cfg = ModelConfig(
        vocab_size=config_dict['vocab_size'],
        hidden_dim=config_dict['hidden_dim'],
        num_layers=config_dict['num_layers'],
        num_heads=config_dict['num_heads'],
        num_kv_heads=config_dict['num_kv_heads'],
        initial_ffn_dim=config_dict['initial_ffn_dim'] # 完美兼容任何拓展后的维度！
    )
    
    model = GrowableLLM(model_cfg)
    print(f"[*] 正在挂载真实权重: {os.path.basename(selected_pt)}")
    model.load_state_dict(torch.load(selected_pt, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval() # 开启评估模式

    # 加载密码本 (Tokenizer)
    tokenizer_path = config_dict.get('tokenizer_path')
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        print(f"{Color.RED}❌ 找不到本地 Tokenizer ({tokenizer_path})！{Color.RESET}")
        sys.exit(1)
        
    print(f"[*] 正在挂载专属密码本...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # 尝试获取 ChatML 的结束符 <|im_end|>，如果没有就用默认的 eos_token
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id == tokenizer.unk_token_id or im_end_id is None:
        im_end_id = tokenizer.eos_token_id

    # =========================================================
    # 3. 沉浸式多轮对话循环
    # =========================================================
    clear_screen()
    print_header(f"🚀 GrowableLLM 灵魂链接已建立 [{os.path.basename(selected_pt)}]")
    print(f"{Color.CYAN}💡 提示：输入 'clear' 清空上下文记忆，输入 'exit' 退出。{Color.RESET}\n")

    # 对话记忆池
    chat_history = []
    
    while True:
        try:
            user_input = input(f"{Color.GREEN}User > {Color.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Color.CYAN}👋 断开链接。{Color.RESET}")
            break
            
        if user_input.lower() in ['exit', 'quit']:
            print(f"{Color.CYAN}👋 断开链接。{Color.RESET}")
            break
        elif user_input.lower() == 'clear':
            chat_history.clear()
            print(f"{Color.YELLOW}[记忆已清空]{Color.RESET}\n")
            continue
        elif not user_input:
            continue

        # 🌟 核心：自动为用户输入穿上 ChatML 剧本外壳！
        # 1. 铺垫系统设定
        prompt = "<|im_start|>system\n你是一个高情商、乐于助人的AI助手。<|im_end|>\n"
        
        # 2. 拼接历史记忆 (多轮对话的关键)
        for history_user, history_bot in chat_history:
            prompt += f"<|im_start|>user\n{history_user}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{history_bot}<|im_end|>\n"
            
        # 3. 放入本次的提问，并以 assistant 开头，引导模型接话！
        prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

        # 转码送入 GPU
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        input_length = input_ids.shape[1]

        # 如果历史太长导致超过模型限制，做个安全拦截
        if input_length > model_cfg.max_seq_len - 256:
            print(f"{Color.RED}⚠️ 记忆超载！请使用 'clear' 指令清空记忆。{Color.RESET}\n")
            continue

        print(f"{Color.CYAN}GrowableLLM > {Color.RESET}", end="", flush=True)

        # 🚀 触发你的 model.py 里的 generate 函数
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                eos_token_id=im_end_id,  # 告诉模型遇到 <|im_end|> 就必须闭嘴
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )

        # 剥离掉我们输入的 Prompt，只截取模型新生成的那部分 ID
        new_ids = output_ids[0][input_length:]
        
        # 将生成的数字 ID 解码回人类文字
        response_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        
        print(f"{response_text}\n")
        
        # 把这次问答存入记忆池
        chat_history.append((user_input, response_text))

if __name__ == "__main__":
    run_chat_console()