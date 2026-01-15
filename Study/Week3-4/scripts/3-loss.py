import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= 配置区域 =================
# 你的模型输出目录 (里面应该有 trainer_state.json 或 checkpoint 文件夹)
MODEL_DIR = "./results/model" 
# 或者指向具体的 checkpoint，例如: "./results/model/checkpoint-500"
# ===========================================

def plot_clean_curve(model_dir):
    # 1. 寻找日志文件
    log_file = os.path.join(model_dir, "trainer_state.json")
    
    # 如果根目录没找到，试着找找最新的 checkpoint 文件夹
    if not os.path.exists(log_file):
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            log_file = os.path.join(model_dir, checkpoints[-1], "trainer_state.json")
    
    if not os.path.exists(log_file):
        print(f"错误：在 {model_dir} 及其子目录中找不到 trainer_state.json 日志文件。")
        return

    print(f"正在读取日志: {log_file}")
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    log_history = data['log_history']
    df = pd.DataFrame(log_history)

    # 2. 清洗“双重线”逻辑 (保留最后一次完整训练)
    if len(df) > 1:
        # 找到 epoch 变小的地方（重启点）
        restart_points = df[df['epoch'].diff() < 0].index.tolist()
        if restart_points:
            last_restart = restart_points[-1]
            print(f"检测到 {len(restart_points)} 次重跑记录，正在截取最后一次训练数据...")
            df = df.iloc[last_restart:].reset_index(drop=True)

    # 3. 提取数据
    train_logs = df[df['loss'].notna() & df['eval_loss'].isna()]
    eval_logs = df[df['eval_loss'].notna()]

    # 4. 绘图
    plt.figure(figsize=(12, 5))

    # Loss 子图
    plt.subplot(1, 2, 1)
    plt.plot(train_logs['epoch'], train_logs['loss'], label='Training Loss', alpha=0.9)
    if not eval_logs.empty:
        plt.plot(eval_logs['epoch'], eval_logs['eval_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve (Cleaned)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Metrics 子图
    plt.subplot(1, 2, 2)
    if not eval_logs.empty:
        if 'eval_f1' in eval_logs.columns:
            plt.plot(eval_logs['epoch'], eval_logs['eval_f1'], label='F1', marker='.')
        if 'eval_mcc' in eval_logs.columns:
            plt.plot(eval_logs['epoch'], eval_logs['eval_mcc'], label='MCC', marker='.')
        if 'eval_acc' in eval_logs.columns:
            plt.plot(eval_logs['epoch'], eval_logs['eval_acc'], label='Accuracy', marker='.')
    
    plt.xlabel('Epoch')
    plt.title('Evaluation Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(model_dir, "training_curves_fixed.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"修复后的图片已保存至: {output_path}")

if __name__ == "__main__":
    plot_clean_curve(MODEL_DIR)