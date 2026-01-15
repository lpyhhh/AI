#!/usr/bin/env python3
"""
input：真实数据集序列文件
output：预测概率 + 注意力热力图可视化 混淆矩阵

怎么看结果：
1 看图片的颜色深浅，颜色越深表示模型越关注该位置的氨基酸残基
2 如果高亮区域对应 RCR motif (Rolling Circle Replication) 或者 Helicase domain 等保守结构域
（会在print输出）
重点看混淆矩阵的第二行：正样本，看看模型能找出多少真实的正样本，如果有比较多的误报FN，可以考虑后续用更大模型或者更多数据进行微调
第一行：负样本，看看模型能剔除多少真实的负样本，如果误报FP比较多，说明模型区分负样本的能力不够强，可以考虑增加负样本数据进行训练
"""
import argparse
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import EsmTokenizer, EsmForSequenceClassification
from peft import PeftModel
from sklearn.metrics import classification_report, matthews_corrcoef, confusion_matrix

# ================= 配置区域 (请修改这里) =================
# 1. 你的真实数据集路径
INPUT_FASTA = "/home/ec2-user/project/results/test/true_verify/data/true_verify.fasta" 

# 2. 模型路径
LORA_PATH = "./results/model/final_lora_model"
BASE_MODEL = "facebook/esm2_t33_650M_UR50D"

# 3. 输出目录
OUTPUT_DIR = "./results/5-inference_vis"

# 4. 可视化数量：你想画多少张图？
# 建议：画 3 张预测正确的正样本，看看它关注哪里
NUM_VISUALIZATIONS = 3 
# ======================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

class RealDataset(Dataset):
    def __init__(self, fasta_file, tokenizer, max_len=1024):
        self.sequences = []
        self.labels = []
        self.ids = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        print(f"Loading data from {fasta_file}...")
        
        # 统计 ID 里的标签
        for record in SeqIO.parse(fasta_file, "fasta"):
            self.ids.append(record.id)
            self.sequences.append(str(record.seq))
            
            # --- 解析标签 (根据你的描述：ID里包含POS或NEG) ---
            # 如果你的真实数据ID里没有标签，这里会默认为 -1 (未知)
            upper_id = record.id.upper()
            if "POS" in upper_id:
                self.labels.append(1)
            elif "NEG" in upper_id:
                self.labels.append(0)
            else:
                self.labels.append(-1) # 未知标签
                
        print(f"已加载: {len(self.sequences)} 条序列")
        print(f"标签分布: Positive(1)={self.labels.count(1)}, Negative(0)={self.labels.count(0)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 注意：这里我们返回原始数据，padding 在 collate_fn 里做
        # 这样方便可视化时知道真实长度
        return {
            'seq': self.sequences[idx],
            'label': self.labels[idx],
            'id': self.ids[idx]
        }

def collate_fn(batch, tokenizer):  # <--- 新增 tokenizer 参数
    # 重新在 collate 里做 tokenization 比较稳妥
    sequences = [x['seq'] for x in batch]
    labels = [x['label'] for x in batch]
    ids = [x['id'] for x in batch]
    
    # 使用传入的 tokenizer
    encodings = tokenizer(sequences, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': torch.tensor(labels),
        'ids': ids,
        'raw_sequences': sequences
    }

def plot_attention(seq_str, attention_matrix, seq_id, save_path):
    """
    绘制注意力热力图
    seq_str: 原始氨基酸序列
    attention_matrix: shape (seq_len, seq_len) -> 这里的 seq_len 是 token 后的长度
    """
    # 获取 tokens (ESM tokenizer 会加 <cls> 和 <eos>)
    tokens = ['<cls>'] + list(seq_str) + ['<eos>']
    
    # 截取注意力矩阵的有效部分 (去除 padding)
    # attention_matrix 是 (padded_len, padded_len)
    valid_len = len(tokens)
    if valid_len > attention_matrix.shape[0]:
        valid_len = attention_matrix.shape[0] # 防止截断错误
        tokens = tokens[:valid_len]
        
    # 我们只关心 <CLS> Token (索引0) 对其他所有位置的关注度
    # 取出第一行: attention[0, :]
    cls_attention = attention_matrix[0, :valid_len].cpu().numpy()
    
    # 绘图
    plt.figure(figsize=(max(12, valid_len/4), 3)) # 根据长度动态调整宽度
    
    # 画热力图 (Reshape 成 1行 多列)
    sns.heatmap(
        cls_attention.reshape(1, -1),
        xticklabels=tokens,
        yticklabels=['<CLS> Focus'],
        cmap="YlGnBu", # 黄-绿-蓝 配色，颜色越深越重要
        cbar=True,
        square=False
    )
    
    plt.title(f"Attention Map: Which residues decide the class?\nID: {seq_id}")
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    # 1. 准备环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    global tokenizer # 方便 collate_fn 使用
    tokenizer = EsmTokenizer.from_pretrained(BASE_MODEL)
    
    # 2. 加载模型 (开启注意力输出)
    print("Loading Model...")
    base_model = EsmForSequenceClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=2, 
        output_attentions=True # <--- 关键！开启上帝视角
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.to(device)
    model.eval()
    
    # 3. 数据加载
    dataset = RealDataset(INPUT_FASTA, tokenizer)
    # 自定义 collate 需要把 tokenizer 传进去，这里简化写法，直接在循环里处理或者使用 lambda
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=False,
        # 使用 lambda 将 tokenizer 传进去
        collate_fn=lambda x: collate_fn(x, tokenizer) 
    )
    
    # 4. 推理
    all_preds = []
    all_labels = []
    all_probs = []
    ids_list = []
    
    vis_count = 0
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            ids = batch['ids']
            raw_seqs = batch['raw_sequences']
            
            # 显式添加 output_attentions=True
            outputs = model(input_ids=input_ids, attention_mask=mask, output_attentions=True)
            
            # 获取预测结果
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1] # 正类概率
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            ids_list.extend(ids)
            
            # --- 可视化逻辑 ---
            if vis_count < NUM_VISUALIZATIONS:
                # 提取最后一层注意力: (batch, num_heads, seq_len, seq_len)
                # 我们取平均头 (average over heads) -> (batch, seq_len, seq_len)
                last_layer_att = outputs.attentions[-1].mean(dim=1)
                
                for i in range(len(ids)):
                    # 只画预测为正样本(1) 且 真实标签也是正样本(1) 的高质量图
                    # 或者你想看它为什么把负样本(0)错判为正样本(1)
                    if preds[i] == 1 and vis_count < NUM_VISUALIZATIONS:
                        vis_save_path = os.path.join(OUTPUT_DIR, f"attention_{ids[i].replace('|','_')}.png")
                        plot_attention(raw_seqs[i], last_layer_att[i], ids[i], vis_save_path)
                        vis_count += 1
                        print(f"  -> Saved attention map for {ids[i]}")

    # 5. 结果统计
    df = pd.DataFrame({
        "ID": ids_list,
        "True_Label": all_labels,
        "Predicted": all_preds,
        "Prob_Pos": all_probs
    })
    
    csv_path = os.path.join(OUTPUT_DIR, "true-prediction_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存至: {csv_path}")
    
    # 6. 打印评估报告 (只针对有标签的数据)
    # 过滤掉标签为 -1 的数据
    valid_df = df[df['True_Label'] != -1]
    
    if len(valid_df) > 0:
        print("\n" + "="*40)
        print("评估报告 (17410正 : 90负)")
        print("="*40)
        
        y_true = valid_df['True_Label']
        y_pred = valid_df['Predicted']
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        print(f"Confusion Matrix:")
        print(f"                 预测负(0)    预测正(1)")
        print(f"真实负(0) [90]   {tn:<10}   {fp:<10}")
        print(f"真实正(1) [1.7w] {fn:<10}   {tp:<10}")
        print("-" * 40)
        
        print("关键指标解读:")
        print(f"1. 召回率 (Recall/Sensitivity): {tp/(tp+fn):.4f} (正样本找出了多少？)")
        print(f"2. 特异性 (Specificity):        {tn/(tn+fp):.4f} (负样本剔除得怎么样？)")
        print(f"3. 总体 MCC:                    {matthews_corrcoef(y_true, y_pred):.4f}")

if __name__ == "__main__":
    main()