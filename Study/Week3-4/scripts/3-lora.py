#!/usr/bin/env python3
"""
input：序列文件
output：训练好的 LoRA 模型 + 训练曲线图
"""
import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt # 导入绘图库
from Bio import SeqIO
from datasets import Dataset
from transformers import (
    EsmTokenizer,
    EsmForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

# ================= 1. 配置与路径 =================
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
TRAIN_FASTA = "/home/ec2-user/project/results/data/input/train.fasta"
VAL_FASTA = "/home/ec2-user/project/results/data/input/val.fasta"
OUTPUT_DIR = "./results/model"
LOG_DIR = "./results/logs" # 新增日志目录

tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)

# ================= 2. 数据处理函数 (保持不变) =================
def parse_fasta(file_path):
    sequences = []
    labels = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
        if 'POS' in record.id:
            labels.append(1)
        elif 'NEG' in record.id:
            labels.append(0)
        else:
            labels.append(0)
    return Dataset.from_dict({'sequence':sequences,"label":labels})

def tokenize_function(examples):
    return tokenizer(examples["sequence"], truncation=True, max_length=1024)

print("Loading datasets...")
raw_train_ds = parse_fasta(TRAIN_FASTA)
raw_val_ds = parse_fasta(VAL_FASTA)

print("开始分词处理...")
tokenized_train = raw_train_ds.map(tokenize_function, batched=True)
tokenized_val = raw_val_ds.map(tokenize_function, batched=True)

tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val = tokenized_val.rename_column("label", "labels")
tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ================= 3. 模型与 LoRA 配置 =================
print("Initializing Model and LoRA...")
model = EsmForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2,
    cache_dir="./model"
)

lora_config = LoraConfig(
    r=0,#该
    lora_alpha=32,
    target_modules=["query", "key", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================= 4. 训练与指标配置 =================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "mcc": matthews_corrcoef(labels, predictions),
        "f1": f1_score(labels, predictions),
        "acc": accuracy_score(labels, predictions)
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOG_DIR,           # 指定 TensorBoard 日志目录
    report_to="tensorboard",       # 【关键修改】开启 TensorBoard
    
    eval_strategy="epoch",
    save_strategy="epoch",         # 每个 epoch 保存一次 checkpoint
    save_total_limit=3,            # 【新增】只保留最近的 3 个 checkpoint，防止硬盘撑爆
    
    learning_rate=5e-5,#该
    per_device_train_batch_size=8,#该
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=4,
    fp16=True,
    num_train_epochs=10,
    weight_decay=0.05,#该
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=10,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ================= 5. 开始训练 (带断点续训检测) =================
print("\n" + "="*60)
print("开始训练")
print("="*60)

# 【关键修改】检查是否存在 Checkpoint
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint")]
    if len(checkpoints) > 0:
        # 找到最新的 checkpoint
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        last_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
        print(f"检测到中断的训练，将从检查点继续: {last_checkpoint}")

# 如果检测到 checkpoint，传入 resume_from_checkpoint=True
trainer.train(resume_from_checkpoint=True if last_checkpoint else None)

# 保存最终模型
trainer.save_model(f"{OUTPUT_DIR}/final_lora_model")
print(f"Model saved to {OUTPUT_DIR}/final_lora_model")

# ================= 6. 自动绘制曲线图 (新增功能) =================
def plot_history(log_history, save_path):
    """
    input: 训练日志列表, 保存路径
    output: 保存训练曲线图到指定路径
    改进点: 自动检测并移除重复的旧日志（根据 Epoch 突然回跳来判断）
    """
    print("正在绘制训练曲线...")
    
    # 转换为 DataFrame
    df = pd.DataFrame(log_history)
    
    # --- 【新增】清洗数据逻辑 ---
    # 如果发现 epoch 变小了（比如从 2.5 变回 0.1），说明发生了重跑
    # 我们只保留最后一次“从头跑到尾”的记录
    if len(df) > 1:
        # 找到 epoch 突然变小的地方（断崖点）
        # .diff() 计算差值，如果差值 < 0，说明时间倒流了
        restart_points = df[df['epoch'].diff() < 0].index.tolist()
        
        if restart_points:
            last_restart_index = restart_points[-1]
            print(f"检测到历史残留日志，正在截取最后一次训练记录 (从索引 {last_restart_index} 开始)...")
            df = df.iloc[last_restart_index:].reset_index(drop=True)
    # ---------------------------

    # 提取训练集 Loss (有 loss 且没有 eval_loss 的行)
    train_logs = df[df['loss'].notna() & df['eval_loss'].isna()]
    # 提取验证集 Metrics (有 eval_loss 的行)
    eval_logs = df[df['eval_loss'].notna()]
    
    if len(train_logs) == 0:
        print("日志数据不足，跳过绘图。")
        return

    plt.figure(figsize=(12, 5))
    
    # 子图 1: Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_logs['epoch'], train_logs['loss'], label='Training Loss', alpha=0.8) # 加点透明度
    if len(eval_logs) > 0:
        plt.plot(eval_logs['epoch'], eval_logs['eval_loss'], label='Validation Loss', linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图 2: 指标曲线
    plt.subplot(1, 2, 2)
    has_metrics = False
    if len(eval_logs) > 0:
        if 'eval_f1' in eval_logs.columns:
            plt.plot(eval_logs['epoch'], eval_logs['eval_f1'], label='F1 Score', marker='.')
            has_metrics = True
        if 'eval_mcc' in eval_logs.columns:
            plt.plot(eval_logs['epoch'], eval_logs['eval_mcc'], label='MCC', marker='.')
            has_metrics = True
        if 'eval_acc' in eval_logs.columns:
            plt.plot(eval_logs['epoch'], eval_logs['eval_acc'], label='Accuracy', marker='.')
            has_metrics = True
            
    if has_metrics:
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Evaluation Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Evaluation Metrics Found', ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"训练曲线图已保存至: {save_path}")

# 执行绘图
plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plot_history(trainer.state.log_history, plot_path)
# trainer.state.log_history 训练日志记录