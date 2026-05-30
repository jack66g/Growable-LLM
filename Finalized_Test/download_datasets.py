import os
from datasets import load_dataset

def download_and_save_datasets():
    # ============================================================
    # 配置保存路径
    # ============================================================
    BASE_DIR = "growable_llm_data"
    MAGICODER_DIR = os.path.join(BASE_DIR, "magicoder_110k")
    EVOL_CODE_DIR = os.path.join(BASE_DIR, "evol_code_80k")
    
    os.makedirs(BASE_DIR, exist_ok=True)

    print("\n===================================================")
    print("🚀 开始拉取 Phase 1 逻辑推理数据集 (Magicoder)")
    print("===================================================")
    # 拉取 Magicoder-Evol-Instruct-110K
    # 包含了极高质量的 OSS-Instruct 和 Evol-Instruct 混合逻辑推演数据
    magicoder_ds = load_dataset("ISE-UIUC/Magicoder-Evol-Instruct-110K", split="train")
    
    print(f"✅ 成功下载 Magicoder，数据量: {len(magicoder_ds)} 条")
    print(f"📊 数据字段: {magicoder_ds.column_names}")
    
    # 展示第一条数据感受一下质量
    print("\n[Preview - Magicoder Sample 0]")
    print(f"Instruction: {magicoder_ds[0].get('instruction', '')[:100]}...")
    
    # 序列化保存到本地，方便 DataLoader 后续光速加载
    magicoder_ds.save_to_disk(MAGICODER_DIR)
    print(f"💾 已保存至本地磁盘: {MAGICODER_DIR}")

    print("\n===================================================")
    print("🚀 开始拉取 Phase 2 纯代码语法数据集 (Evol-Instruct-Code)")
    print("===================================================")
    # 拉取 Evol-Instruct-Code-80k-v1
    # 纯正的代码生成、重构和 Debug 训练语料
    evol_code_ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
    
    print(f"✅ 成功下载 Evol-Instruct-Code，数据量: {len(evol_code_ds)} 条")
    print(f"📊 数据字段: {evol_code_ds.column_names}")
    
    # 展示第一条数据
    print("\n[Preview - Evol-Code Sample 0]")
    print(f"Instruction: {evol_code_ds[0].get('instruction', '')[:100]}...")
    
    evol_code_ds.save_to_disk(EVOL_CODE_DIR)
    print(f"💾 已保存至本地磁盘: {EVOL_CODE_DIR}")

    print("\n===================================================")
    print("🎉 所有粮草已备齐！")
    print("后续在 DataLoader 中只需使用 load_from_disk() 即可瞬间读取。")
    print("===================================================\n")

if __name__ == "__main__":
    download_and_save_datasets()