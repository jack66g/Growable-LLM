import os
import sys
import time
import json

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

def get_file_line_count(file_path):
    """极其快速的统计文件行数（数据量）的方法"""
    try:
        with open(file_path, 'rb') as f:
            return sum(1 for _ in f)
    except:
        return 0

def run_train_configurator(base_model_config=None):
    """
    核心功能：扫描数据舱，配置动态扩容参数，设置超参，选择 Defrag 策略。
    """
    print_header("⚙️ 训练数据装载与动态扩容配置 (Training Configurator)")
    
    if base_model_config:
        print(f"[{Color.GREEN}System{Color.RESET}] 已成功接收底座模型参数 (Hidden: {base_model_config.get('hidden_dim')}, FFN: {base_model_config.get('initial_ffn_dim')})")
    else:
        print(f"[{Color.YELLOW}Warning{Color.RESET}] 未接收到底座参数，当前为独立配置模式。")

    train_config = {}

    # =========================================================
    # 【1. 数据选择】扫描本地数据舱
    # =========================================================
    data_dir = "./data_cabin"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\n{Color.BOLD}【1. 数据选择】{Color.RESET}已扫描到 {data_dir}/ 目录下的数据集：")
    
    data_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    
    # 🌟 核心修改：移除模拟数据，找不到真实数据直接报错拦截
    if not data_files:
        print(f"{Color.RED}❌ 数据舱为空！请先将训练数据 (.jsonl等) 放入 {data_dir} 文件夹中。{Color.RESET}")
        time.sleep(1)
        return None
        
    file_info = []
    for i, f_name in enumerate(data_files):
        f_path = os.path.join(data_dir, f_name)
        f_size = get_file_line_count(f_path)
        file_info.append({"name": f_name, "path": f_path, "count": f_size})
        print(f"  [{i+1}] {f_name.ljust(25)} (数量: {Color.YELLOW}{f_size:,}{Color.RESET} 条)")
        
    d_choice = input(f"\n> 请选择训练数据 (输入序号 1-{len(data_files)}): ").strip()
    if d_choice.isdigit() and 1 <= int(d_choice) <= len(file_info):
        selected_data = file_info[int(d_choice)-1]
    else:
        selected_data = file_info[0]
        
    train_config['data_name'] = selected_data['name'] # 保存名字，方便日志打印
    train_config['data_path'] = selected_data['path']
    data_count = selected_data['count']
    print(f"[*] 已选定: {Color.GREEN}{selected_data['name']}{Color.RESET}")

    # =========================================================
    # 【2. 动态扩容配置】
    # =========================================================
    print(f"\n{Color.BOLD}【2. 动态扩容配置】{Color.RESET}")
    print(f"检测到当前数据量为: {Color.YELLOW}{data_count:,}{Color.RESET} 条。")
    print("系统默认扩容规则 (容量瓶颈定律)：")
    print("  - 1万以内: +64维  | 5万以内: +128维")
    print("  - 10万以内: +256维 | 50万以内: +512维 | 百万级: +1024维")
    print(f"  {Color.CYAN}(* 若为强推理/数学逻辑，建议维数 × 2){Color.RESET}\n")

    rec_dim = 64
    if data_count <= 10000: rec_dim = 64
    elif data_count <= 50000: rec_dim = 128
    elif data_count <= 100000: rec_dim = 256
    elif data_count <= 500000: rec_dim = 512
    else: rec_dim = 1024

    print(f"当前数据量 ({data_count/10000:.1f}w) 适用基础推荐：{Color.GREEN}+{rec_dim}维{Color.RESET}。")
    print(f"  [1] 采用系统默认推荐 ({Color.GREEN}+{rec_dim}维{Color.RESET})")
    print(f"  [2] 开启强推理双倍模式 ({Color.RED}+{rec_dim * 2}维{Color.RESET})")
    print("  [3] 手动输入自定义扩容维度")
    
    dim_choice = input(f"> 请选择扩容策略 (1-3): ").strip()
    
    if dim_choice == '2':
        train_config['expand_dim'] = rec_dim * 2
    elif dim_choice == '3':
        user_dim = input(f"> 请输入自定义扩容维度 (如 512): ").strip()
        train_config['expand_dim'] = int(user_dim) if user_dim.isdigit() else rec_dim
    else:
        train_config['expand_dim'] = rec_dim

    # =========================================================
    # 【3. 训练超参设置】
    # =========================================================
    print(f"\n{Color.BOLD}【3. 训练超参设置】{Color.RESET}")
    print("  [1] 使用快速默认参数 (Context: 4096, Batch: 8, LR: 2e-4, Save: 1000)")
    print("  [2] 手动自定义参数 (直接回车保留默认值)")
    
    hp_choice = input(f"> 请选择 (1-2): ").strip()
    
    if hp_choice == '2':
        print(f"\n{Color.CYAN}--- 自定义参数 (留空直接回车即为默认) ---{Color.RESET}")
        
        ctx_input = input(f"🔹 上下文长度 Context [默认 4096]: ").strip()
        ctx_len = int(ctx_input) if ctx_input.isdigit() else 4096
        if ctx_len > 8192:
            print(f"{Color.RED}⚠️ [安全拦截] 超过 8192 极易 OOM，已强制回退至 8192。{Color.RESET}")
            ctx_len = 8192
        train_config['max_context'] = ctx_len
        
        batch_input = input(f"🔹 训练批次 Batch Size [默认 8]: ").strip()
        train_config['batch_size'] = int(batch_input) if batch_input.isdigit() else 8
        
        lr_input = input(f"🔹 学习率 LR [默认 2e-4]: ").strip()
        try:
            train_config['learning_rate'] = float(lr_input) if lr_input else 2e-4
        except ValueError:
            train_config['learning_rate'] = 2e-4
        
        save_input = input(f"🔹 保存步数 Save Steps [默认 1000]: ").strip()
        train_config['save_steps'] = int(save_input) if save_input.isdigit() else 1000
    else:
        train_config['max_context'] = 4096
        train_config['batch_size'] = 8
        train_config['learning_rate'] = 2e-4
        train_config['save_steps'] = 1000

    # =========================================================
    # 【4. 核心对齐与抗遗忘策略】
    # =========================================================
    print(f"\n{Color.BOLD}【4. 核心对齐与抗遗忘策略】🌟{Color.RESET}")
    print("请选择模型在训练完成后的碎片整理 (Defrag) 方案：")
    print(f"  [1] 无 Defrag (常规微调)")
    print(f"      {Color.CYAN}- 解释: 速度最快，但会引发严重的灾难性遗忘。{Color.RESET}")
    print(f"  [2] Defrag (仅用当前数据)")
    print(f"      {Color.CYAN}- 解释: 极低显存开销，仅用当前新知识微调顶层通道，抗遗忘效果良好。{Color.RESET}")
    print(f"  [3] Defrag + 仅回放 2 条历史锚点 {Color.RED}[🔥 极力推荐]{Color.RESET}")
    print(f"      {Color.CYAN}- 解释: 结合历史记忆池抽取 2 条锚点数据融合，物理隔绝特征干扰，遗忘率极低！{Color.RESET}")
    
    defrag_choice = input(f"> 请选择策略 (1-3): ").strip()
    
    defrag_map = {"1": "NONE", "2": "CURRENT_ONLY", "3": "REPLAY_2"}
    train_config['defrag_strategy'] = defrag_map.get(defrag_choice, "REPLAY_2")
    defrag_display = {"NONE": "无 Defrag", "CURRENT_ONLY": "当前数据融合", "REPLAY_2": "Defrag + 回放 2 条"}[train_config['defrag_strategy']]

    # =========================================================
    # 汇总并落地为缓存文件 (Payload)
    # =========================================================
    final_payload = {
        "base_model": base_model_config if base_model_config else {},
        "training": train_config
    }

    print_header("✅ 配置确认完毕！")
    print(f" - 目标数据   : {Color.GREEN}{train_config['data_name']}{Color.RESET}")
    print(f" - 物理扩容   : {Color.GREEN}+{train_config['expand_dim']} 维{Color.RESET}")
    print(f" - 上下文长   : {Color.GREEN}{train_config['max_context']}{Color.RESET}")
    print(f" - Batch Size : {Color.GREEN}{train_config['batch_size']}{Color.RESET}")
    print(f" - 学习率(LR) : {Color.GREEN}{train_config['learning_rate']}{Color.RESET}")
    print(f" - 保存步数   : {Color.GREEN}{train_config['save_steps']} 步{Color.RESET}")
    print(f" - 抗遗忘策略 : {Color.GREEN}[{defrag_display}]{Color.RESET}")
    
    confirm = input(f"\n{Color.RED}是否将配置指令发往底层引擎？(y/n): {Color.RESET}").strip().lower()
    
    if confirm == 'y':
        # 将所有参数写入 JSON 缓存文件
        cache_file = "train_payload.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=4, ensure_ascii=False)
            
        print(f"\n{Color.CYAN}🚀 [Engine Log] 配置已打包为 {cache_file} 缓存文件，等待训练引擎读取！{Color.RESET}")
        time.sleep(1)
        return final_payload
    else:
        print(f"\n{Color.YELLOW}已取消，参数被丢弃。{Color.RESET}")
        return None

if __name__ == "__main__":
    # 🌟 核心修改：直接运行，不再有任何硬编码的假数据
    run_train_configurator()