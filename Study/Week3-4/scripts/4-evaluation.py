#!/usr/bin/env python3
"""
input：测试序列文件
output：
    真实样本的预测标签概率和结果 保存为CSV文件
"""
import argparse
import os
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import EsmTokenizer, EsmForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report

# ================= 1. 参数解析配置 =================
def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation script for ESM-2 with LoRA adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必需参数
    parser.add_argument(
        "-i", "--input_fasta", 
        required=True, 
        help="Path to the test FASTA file (e.g., test.fasta)"
    )
    parser.add_argument(
        "-m", "--lora_model", 
        required=True, 
        help="Path to the trained LoRA model directory (e.g., ./results/model/final_lora_model)"
    )

    # 可选参数 (有默认值)
    parser.add_argument(
        "-o", "--output_csv", 
        default="./results/test_predictions.csv", 
        help="Path to save the prediction results CSV"
    )
    parser.add_argument(
        "--base_model", 
        default="facebook/esm2_t33_650M_UR50D", 
        help="Hugging Face base model name (must match training)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=128, 
        help="Batch size for inference (adjust based on GPU VRAM)"
    )
    parser.add_argument(
        "--max_len", 
        type=int, 
        default=1024, 
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--cache_dir", 
        default="./model", 
        help="Directory to cache downloaded base models"
    )
    #问题：1 我用的应该是微调后的模型，为什么要加载原始模型？
    return parser.parse_args()

# ================= 2. 数据处理类 =================
class ProteinDataset(Dataset):
    def __init__(self, fasta_file, tokenizer, max_len=1024):
        #文件处理
        #input：fa
        #output: 为什么没有输出？ 在getitem中输出。init只加载数据，即可以加载数据进行处理，
        self.sequences = []
        self.labels = []
        self.ids = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        print(f"Loading data from {fasta_file}...")
        
        # 检查文件是否存在
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"Input file not found: {fasta_file}")

        for record in SeqIO.parse(fasta_file, "fasta"):
            #解析序列
            self.ids.append(record.id)
            self.sequences.append(str(record.seq))
            
            # --- 解析标签逻辑 (保持原有逻辑) ---
            if "POS_" in record.id:
                self.labels.append(1)
            elif "NEG_" in record.id:
                self.labels.append(0)
            else:
                self.labels.append(0)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        #对序列进行token
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            seq,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,#问题：这里的序列长度是如何把parser传入过来的？
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(), #多维向量 拆成一维
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'id': self.ids[idx]
        }
        #问题：后面如何调用？也就是代码逻辑问题

# ================= 3. 主流程 =================
def main():
    # 获取命令行参数
    args = get_args()

    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Batch Size: {args.batch_size}")

    # 2. 准备输出目录
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 3. 加载 Tokenizer
    print(f"Loading Tokenizer ({args.base_model})...")
    tokenizer = EsmTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)

    # 4. 加载模型
    print("Loading Base Model...")
    base_model = EsmForSequenceClassification.from_pretrained(
        args.base_model, 
        num_labels=2, 
        cache_dir=args.cache_dir
    )
    
    print(f"Loading LoRA adapters from {args.lora_model}...")
    try:
        model = PeftModel.from_pretrained(base_model, args.lora_model)
    except Exception as e:
        print(f"Error loading LoRA model: {e}")
        print("Make sure the path points to the folder containing adapter_config.json and adapter_model.bin")
        return

    model.to(device)
    model.eval()

    # 5. 准备数据
    test_dataset = ProteinDataset(args.input_fasta, tokenizer, max_len=args.max_len)#直接返回模型需要的字典型数据
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 6. 开始推理
    print("Running Inference...")
    all_preds = []
    all_labels = []
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            ids = batch['id']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 计算概率 (Softmax)
            probs = torch.softmax(logits, dim=1)[:, 1] # 取正类(1)的概率
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_ids.extend(ids)

    # 7. 计算指标
    valid_indices = [i for i, x in enumerate(all_labels) if x != -1]
    
    if len(valid_indices) > 0:
        clean_labels = [all_labels[i] for i in valid_indices]
        clean_preds = [all_preds[i] for i in valid_indices]
        
        print("\n" + "="*30)
        print("TEST RESULTS")
        print("="*30)
        acc = accuracy_score(clean_labels, clean_preds)
        f1 = f1_score(clean_labels, clean_preds)
        mcc = matthews_corrcoef(clean_labels, clean_preds)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"MCC:      {mcc:.4f}")
        print("\nDetailed Report:")
        print(classification_report(clean_labels, clean_preds, target_names=["Negative", "Positive"]))
    else:
        print("\nWarning: No valid labeled data found (all labels were -1). Skipping metrics calculation.")

    # 8. 保存结果
    df = pd.DataFrame({
        "ID": all_ids,
        "True_Label": all_labels,
        "Predicted_Label": all_preds,
        "Positive_Prob": all_probs
    })
    df.to_csv(args.output_csv, index=False)
    print(f"\nPredictions saved to {args.output_csv}")

if __name__ == "__main__":
    main()