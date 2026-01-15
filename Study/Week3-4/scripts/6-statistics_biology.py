#!/usr/bin/env python3
"""
input：先小样本测试，后全量样本测试
output：概率直方图 + t-SNE 可视化 + 3D 注意力权重映射文件

3D 注意力权重映射 的 txt文件怎么生成的？

"""
import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from Bio import SeqIO
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from transformers import EsmTokenizer, EsmForSequenceClassification
from peft import PeftModel

# ================= 1. 参数解析配置 =================
def get_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Analysis: Probability Histograms, t-SNE, and 3D Attention Mapping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必需参数
    parser.add_argument(
        "-i", "--input_fasta", 
        required=True, 
        help="Path to the input FASTA file (e.g., test_dataset.fasta)"
    )
    parser.add_argument(
        "-m", "--lora_model", 
        required=True, 
        help="Path to the trained LoRA model directory"
    )

    # 可选参数
    parser.add_argument(
        "-o", "--output_dir", 
        default="./results/analysis_plots", 
        help="Directory to save analysis plots and data"
    )
    parser.add_argument(
        "--base_model", 
        default="facebook/esm2_t33_650M_UR50D", 
        help="Hugging Face base model name"
    )
    parser.add_argument(
        "--tsne_samples", 
        type=int, 
        default=3000, 
        help="Number of samples to use for t-SNE visualization (to save time)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="Batch size for inference"
    )
    
    return parser.parse_args()

# ================= 2. 数据与工具类 =================
class AnalysisDataset(Dataset):
    def __init__(self, fasta_file):
        self.data = []
        # 简单的正负样本解析逻辑
        print(f"Loading data from {fasta_file}...")
        for record in SeqIO.parse(fasta_file, "fasta"):
            # 根据 ID 判断标签，你可以根据实际情况修改这里
            label = 1 if "POS" in record.id.upper() else 0
            self.data.append({
                "id": record.id,
                "seq": str(record.seq),
                "label": label
            })
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch, tokenizer):
    seqs = [x['seq'] for x in batch]
    labels = [x['label'] for x in batch]
    ids = [x['id'] for x in batch]
    
    inputs = tokenizer(seqs, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    inputs['labels'] = torch.tensor(labels)
    inputs['ids'] = ids
    inputs['raw_seqs'] = seqs
    return inputs

# ================= 3. 核心功能函数 =================
def get_model_outputs(args, device):
    print("1. 加载模型与数据...")
    tokenizer = EsmTokenizer.from_pretrained(args.base_model)
    base_model = EsmForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, output_hidden_states=True, output_attentions=True
    )
    
    print(f"Loading LoRA from {args.lora_model}...")
    model = PeftModel.from_pretrained(base_model, args.lora_model)
    model.to(device)
    model.eval()

    dataset = AnalysisDataset(args.input_fasta)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, # Shuffle以便t-SNE采样随机
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )

    all_probs = []
    all_labels = []
    all_embeddings = [] 
    
    # 用于 3D 映射的示例数据
    sample_for_3d = None 

    print("2. 开始推理提取特征...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=mask, 
                output_hidden_states=True,  # <--- 必须显式开启，否则无法做 t-SNE
                output_attentions=True      # <--- 必须显式开启，否则无法做 3D 映射
            )
            
            # 1. 概率
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch['labels'].numpy())

            # 2. 嵌入向量 (取最后一层 hidden state 的 <CLS>)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            all_embeddings.extend(cls_embeddings)

            # 3. 抓取一个高置信度的正样本用于 3D 分析 (只抓一次)
            if sample_for_3d is None:
                for i, p in enumerate(probs):
                    if p > 0.95 and batch['labels'][i] == 1:
                        att = outputs.attentions[-1][i].mean(dim=0).cpu().numpy()
                        # 去掉CLS和EOS的偏移量
                        raw_len = len(batch['raw_seqs'][i])
                        cls_att = att[0, 1 : raw_len + 1] 
                        
                        sample_for_3d = {
                            "id": batch['ids'][i],
                            "seq": batch['raw_seqs'][i],
                            "attention": cls_att
                        }
                        break
            
            # 内存保护：如果数据量太大，只跑一部分用于可视化
            if len(all_probs) > args.tsne_samples * 1.5:
                print(f"已收集足够样本 ({len(all_probs)})，提前停止推理以节省时间...")
                break

    return np.array(all_probs), np.array(all_labels), np.array(all_embeddings), sample_for_3d

def plot_histogram(probs, labels, output_dir):
    print("3. 绘制概率直方图...")
    plt.figure(figsize=(10, 6))
    
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]
    
    plt.hist(neg_probs, bins=50, alpha=0.6, label='Negative Samples', color='blue')
    plt.hist(pos_probs, bins=50, alpha=0.6, label='Positive Samples', color='red')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, "prob_histogram.png")
    plt.savefig(save_path)
    plt.close()
    print(f"-> 直方图已保存: {save_path}")

def plot_tsne(embeddings, labels, output_dir, max_samples):
    print(f"4. 正在计算 t-SNE (当前样本数: {len(embeddings)})...")
    
    # 限制点数防止太慢
    if len(embeddings) > max_samples:
        print(f"采样至 {max_samples} 个点进行可视化...")
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    # PCA 预降维
    pca = PCA(n_components=min(50, embeddings.shape[1]))
    pca_result = pca.fit_transform(embeddings)
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, verbose=1, perplexity=30)#, n_iter=1000
    tsne_results = tsne.fit_transform(pca_result)
    
    plt.figure(figsize=(10, 10))
    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    df_subset['Label'] = ['Positive' if l==1 else 'Negative' for l in labels]
    
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Label",
        palette={"Positive": "red", "Negative": "blue"},
        data=df_subset,
        legend="full",
        alpha=0.6
    )
    plt.title('t-SNE visualization of ESM-2 Embeddings')
    
    save_path = os.path.join(output_dir, "tsne_embedding.png")
    plt.savefig(save_path)
    plt.close()
    print(f"-> t-SNE图已保存: {save_path}")

def save_pdb_attention(sample_data, output_dir):
    if sample_data is None:
        print("未找到符合条件的正样本用于 3D 映射。")
        return
        
    print(f"5. 生成 3D 映射文件 (ID: {sample_data['id']})...")
    
    seq_path = os.path.join(output_dir, "structure_target.fasta")
    att_path = os.path.join(output_dir, "structure_weights.txt")
    
    with open(seq_path, "w") as f:
        f.write(f">{sample_data['id']}\n{sample_data['seq']}\n")
    
    # 归一化注意力权重到 0-100
    att = sample_data['attention']
    att_norm = (att - att.min()) / (att.max() - att.min()) * 100
    
    np.savetxt(att_path, att_norm, fmt="%.4f")
    
    print(f"-> 序列已保存至: {seq_path}")
    print(f"-> 权重已保存至: {att_path}")

# ================= 4. 主程序 =================
if __name__ == "__main__":
    args = get_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 执行流程
    probs, labels, embeddings, sample_3d = get_model_outputs(args, device)
    plot_histogram(probs, labels, args.output_dir)
    plot_tsne(embeddings, labels, args.output_dir, args.tsne_samples)
    save_pdb_attention(sample_3d, args.output_dir)
    
    print(f"\n所有分析已完成！结果位于: {args.output_dir}")