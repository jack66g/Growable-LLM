import os
import sys
import time
import subprocess 

# 注入我们的独立模块
from base_builder.pull_engine import run_pull_engine
from training_engine.train_configurator import run_train_configurator
# 🌟 导入新的对话模块
from inference.chat_console import run_chat_console

class Color:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # 用于在不同模块间传递模型参数的“上下文变量”
    current_model_config = None

    while True:
        clear_screen()
        print(f"{Color.BOLD}{Color.CYAN}")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║        🚀 GrowableLLM 动态生长控制台 (MLOps CLI v1.0)        ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"{Color.RESET}")
        
        # 智能状态栏，显示具体加载的模型名
        if current_model_config:
            model_name = os.path.basename(current_model_config.get('model_path', 'Unknown'))
            print(f"当前底座状态: {Color.GREEN}✅ 已载入 [{model_name}]{Color.RESET}\n")
        else:
            print(f"当前底座状态: {Color.RED}❌ 未载入 (请先执行步骤 1 提取底座){Color.RESET}\n")
            
        print(f"{Color.BOLD}【系统导航菜单】{Color.RESET}\n")
        print(f"  [{Color.GREEN}1{Color.RESET}] 📥 基座引擎构建 (提取参数、密码本与权重映射)")
        print(f"  [{Color.GREEN}2{Color.RESET}] ⚙️  配置并启动训练 (动态扩容与 Defrag)")
        print(f"  [{Color.GREEN}3{Color.RESET}] 💬 模型对话测试 (交互式推理验证)")
        print(f"  [{Color.RED}0{Color.RESET}] ❌ 退出系统\n")
        print(f"{Color.CYAN}──────────────────────────────────────────────────────────────{Color.RESET}")
        
        choice = input(f"\n{Color.YELLOW}root@growable-llm:~# {Color.RESET}").strip()
        
        if choice == '1':
            # 选项1 返回的是我们之前生成的 config_dict
            result = run_pull_engine()
            if result: 
                current_model_config = result
        
        elif choice == '2':
            # 选项2 将提取到的模型参数传参给训练配置器
            payload = run_train_configurator(base_model_config=current_model_config)
            
            # 自动接管并点火底层的 start_training.py！
            if payload:
                print(f"\n{Color.CYAN}⚙️ 检测到训练配置已生成，是否立即点火启动底层炼丹炉？(y/n){Color.RESET}")
                start_choice = input(f"{Color.YELLOW}> {Color.RESET}").strip().lower()
                
                if start_choice == 'y':
                    print(f"\n{Color.GREEN}🚀 正在将控制权移交给底层训练引擎...{Color.RESET}")
                    time.sleep(1)
                    
                    # 动态获取炼丹炉脚本路径并启动
                    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_engine", "start_training.py")
                    subprocess.run([sys.executable, script_path])
                    
                    print(f"\n{Color.CYAN}🔔 训练进程已结束，控制权已交回主控台。{Color.RESET}")
                    input(f"按回车键返回主菜单...")
        
        elif choice == '3':
            # 🌟 完善：直接调用真实对话引擎
            run_chat_console()
            
        elif choice == '0':
            print(f"\n{Color.CYAN}👋 退出系统。{Color.RESET}")
            sys.exit(0)

if __name__ == "__main__":
    main()