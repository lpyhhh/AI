#!/usr/bin/env python3
"""
Integrated Analysis Tool for ESM-2 LoRA Model Evaluation

功能整合：
1. 基础评估 (来自 4-evaluation.py)
   - 预测结果保存
   - 准确率、F1、MCC指标计算
   
2. 真实数据分析 (来自 5-true_DB.py)
   - 注意力热力图可视化
   - 混淆矩阵分析
   
3. 高级统计分析 (来自 6-statistics_biology.py)
   - 概率分布直方图
   - t-SNE 降维可视化
   - 3D 注意力权重映射

使用示例：
python integrated_analysis.py \
    --mode evaluation \
    --input_fasta test.fasta \
    --lora_model ./results/model/final_lora_model \
    --output_dir ./results/analysis
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
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, 
    classification_report, confusion_matrix
)
from torch.utils.data import DataLoader, Dataset
from transformers import EsmTokenizer, EsmForSequenceClassification
from peft import PeftModel


class ModelAnalyzer:
    """
    统一的模型分析类，整合所有评估功能
    """
    def __init__(self, args):
        """初始化分析器"""
        self.args = args
        self._setup_directories()
        self._setup_device()
        self._load_model()
        
        # 存储结果
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.embeddings = []
        self.ids = []
        self.sequences = []
        self.attentions = []
        
    def _setup_directories(self):
        """创建输出目录"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        print(f"[Setup] Output directory: {self.args.output_dir}")
        
    def _setup_device(self):
        """设置计算设备"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Setup] Using device: {self.device}")
        
    def _load_model(self):
        """加载模型和tokenizer"""
        print("[Model] Loading tokenizer and model...")
        
        # Tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(
            self.args.base_model,
            cache_dir=self.args.cache_dir
        )
        
        # Base Model
        base_model = EsmForSequenceClassification.from_pretrained(
            self.args.base_model,
            num_labels=2,
            output_hidden_states=True,
            output_attentions=True,
            cache_dir=self.args.cache_dir
        )
        
        # Load LoRA
        print(f"[Model] Loading LoRA from {self.args.lora_model}...")
        self.model = PeftModel.from_pretrained(base_model, self.args.lora_model)
        self.model.to(self.device)
        self.model.eval()
        print("[Model] Model loaded successfully!")
        
    # ================= 数据处理 =================
    def load_data(self):
        """加载FASTA数据"""
        print(f"[Data] Loading sequences from {self.args.input_fasta}...")
        
        if not os.path.exists(self.args.input_fasta):
            raise FileNotFoundError(f"Input file not found: {self.args.input_fasta}")
        
        sequences = []
        labels = []
        ids = []
        
        for record in SeqIO.parse(self.args.input_fasta, "fasta"):
            ids.append(record.id)
            sequences.append(str(record.seq))
            
            # 解析标签
            upper_id = record.id.upper()
            if "POS" in upper_id:
                labels.append(1)
            elif "NEG" in upper_id:
                labels.append(0)
            else:
                labels.append(-1)  # 未知标签
                
        print(f"[Data] Loaded {len(sequences)} sequences")
        print(f"[Data] Label distribution: Positive={labels.count(1)}, "
              f"Negative={labels.count(0)}, Unknown={labels.count(-1)}")
        
        return sequences, labels, ids
    
    def _create_dataloader(self, sequences, labels, ids):
        """创建数据加载器"""
        dataset = SimpleDataset(sequences, labels, ids)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=lambda x: self._collate_fn(x)
        )
        return loader
    
    def _collate_fn(self, batch):
        """批处理函数"""
        sequences = [x['seq'] for x in batch]
        labels = [x['label'] for x in batch]
        ids = [x['id'] for x in batch]
        
        encodings = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels),
            'ids': ids,
            'sequences': sequences
        }
    
    # ================= 核心推理 =================
    def run_inference(self):
        """执行模型推理"""
        print("[Inference] Starting inference...")
        
        sequences, labels, ids = self.load_data()
        dataloader = self._create_dataloader(sequences, labels, ids)
        
        # 限制样本数量（用于t-SNE）
        max_samples = self.args.tsne_samples if self.args.mode in ['tsne', 'full'] else len(sequences)
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True
                )
                
                # 提取结果
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                self.predictions.extend(preds)
                self.labels.extend(batch['labels'].numpy())
                self.probabilities.extend(probs)
                self.ids.extend(batch['ids'])
                self.sequences.extend(batch['sequences'])
                
                # 提取嵌入向量（用于t-SNE）
                if self.args.mode in ['tsne', 'full']:
                    cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                    self.embeddings.extend(cls_embeddings)
                
                # 提取注意力（用于可视化）
                if self.args.mode in ['attention', 'full']:
                    last_layer_att = outputs.attentions[-1].mean(dim=1)
                    for i in range(len(batch['ids'])):
                        self.attentions.append({
                            'id': batch['ids'][i],
                            'seq': batch['sequences'][i],
                            'attention': last_layer_att[i].cpu().numpy(),
                            'pred': preds[i],
                            'label': batch['labels'][i].item(),
                            'prob': probs[i]
                        })
                
                sample_count += len(batch['ids'])
                if sample_count >= max_samples:
                    print(f"[Inference] Reached sample limit ({max_samples}), stopping...")
                    break
        
        print(f"[Inference] Completed! Processed {len(self.predictions)} samples")
        
    # ================= 1. 基础评估 (from 4-evaluation.py) =================
    def save_predictions(self):
        """保存预测结果为CSV"""
        print("[Evaluation] Saving predictions...")
        
        df = pd.DataFrame({
            "ID": self.ids,
            "True_Label": self.labels,
            "Predicted_Label": self.predictions,
            "Positive_Prob": self.probabilities
        })
        
        csv_path = os.path.join(self.args.output_dir, "predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"[Evaluation] Predictions saved to {csv_path}")
        
    def compute_metrics(self):
        """计算评估指标"""
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        
        # 过滤有效标签
        valid_mask = np.array(self.labels) != -1
        if not valid_mask.any():
            print("[Warning] No valid labels found for evaluation")
            return
        
        y_true = np.array(self.labels)[valid_mask]
        y_pred = np.array(self.predictions)[valid_mask]
        
        # 基础指标
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_true, y_pred)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"MCC:       {mcc:.4f}")
        
        # 混淆矩阵
        print("\n" + "-"*50)
        print("CONFUSION MATRIX")
        print("-"*50)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        print(f"                 Predicted Neg    Predicted Pos")
        print(f"Actual Neg       {tn:<15} {fp:<15}")
        print(f"Actual Pos       {fn:<15} {tp:<15}")
        
        print("\n关键指标解读:")
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  Recall (灵敏度):     {recall:.4f} - 正样本找出率")
        print(f"  Specificity (特异性): {specificity:.4f} - 负样本剔除率")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))
        
    # ================= 2. 注意力可视化 (from 5-true_DB.py) =================
    def visualize_attention(self):
        """生成注意力热力图"""
        print("[Visualization] Generating attention maps...")
        
        if not self.attentions:
            print("[Warning] No attention data available")
            return
        
        # 选择高置信度正样本进行可视化
        positive_samples = [
            att for att in self.attentions 
            if att['pred'] == 1 and att['prob'] > 0.9
        ]
        
        num_vis = min(self.args.num_attention_plots, len(positive_samples))
        
        for i in range(num_vis):
            sample = positive_samples[i]
            self._plot_attention_heatmap(
                sample['seq'],
                sample['attention'],
                sample['id'],
                sample['prob']
            )
        
        print(f"[Visualization] Generated {num_vis} attention maps")
        
    def _plot_attention_heatmap(self, sequence, attention_matrix, seq_id, prob):
        """绘制单个注意力热力图"""
        # 获取tokens
        tokens = ['<cls>'] + list(sequence) + ['<eos>']
        valid_len = min(len(tokens), attention_matrix.shape[0])
        
        # 提取CLS token的注意力
        cls_attention = attention_matrix[0, :valid_len]
        
        # 绘图
        fig, ax = plt.subplots(figsize=(max(12, valid_len/4), 3))
        
        sns.heatmap(
            cls_attention.reshape(1, -1),
            xticklabels=tokens[:valid_len],
            yticklabels=['CLS Focus'],
            cmap="YlGnBu",
            cbar=True,
            square=False,
            ax=ax
        )
        
        ax.set_title(f"Attention Map | ID: {seq_id} | Prob: {prob:.4f}")
        plt.xticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        safe_id = seq_id.replace('|', '_').replace('/', '_')
        save_path = os.path.join(self.args.output_dir, f"attention_{safe_id}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
    # ================= 3. 概率分布 (from 6-statistics_biology.py) =================
    def plot_probability_histogram(self):
        """绘制预测概率分布直方图"""
        print("[Statistics] Plotting probability histogram...")
        
        probs = np.array(self.probabilities)
        labels = np.array(self.labels)
        
        # 过滤有效标签
        valid_mask = labels != -1
        if not valid_mask.any():
            print("[Warning] No valid labels for histogram")
            return
        
        probs = probs[valid_mask]
        labels = labels[valid_mask]
        
        pos_probs = probs[labels == 1]
        neg_probs = probs[labels == 0]
        
        plt.figure(figsize=(10, 6))
        plt.hist(neg_probs, bins=50, alpha=0.6, label='Negative Samples', color='blue')
        plt.hist(pos_probs, bins=50, alpha=0.6, label='Positive Samples', color='red')
        
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Prediction Confidence Distribution', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.args.output_dir, "prob_histogram.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Statistics] Histogram saved to {save_path}")
        
    # ================= 4. t-SNE 可视化 =================
    def plot_tsne(self):
        """t-SNE降维可视化"""
        print("[Statistics] Computing t-SNE visualization...")
        
        if len(self.embeddings) == 0:
            print("[Warning] No embeddings available for t-SNE")
            return
        
        embeddings = np.array(self.embeddings)
        labels = np.array(self.labels[:len(embeddings)])
        
        # 过滤有效标签
        valid_mask = labels != -1
        if not valid_mask.any():
            print("[Warning] No valid labels for t-SNE")
            return
        
        embeddings = embeddings[valid_mask]
        labels = labels[valid_mask]
        
        # PCA预降维
        print("[t-SNE] Applying PCA pre-processing...")
        pca = PCA(n_components=min(50, embeddings.shape[1]))
        pca_result = pca.fit_transform(embeddings)
        
        # t-SNE降维
        print("[t-SNE] Computing t-SNE (this may take a while)...")
        tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(pca_result)
        
        # 绘图
        plt.figure(figsize=(10, 10))
        df_subset = pd.DataFrame({
            'tsne-2d-one': tsne_results[:, 0],
            'tsne-2d-two': tsne_results[:, 1],
            'Label': ['Positive' if l == 1 else 'Negative' for l in labels]
        })
        
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="Label",
            palette={"Positive": "red", "Negative": "blue"},
            data=df_subset,
            legend="full",
            alpha=0.6
        )
        
        plt.title('t-SNE Visualization of ESM-2 Embeddings', fontsize=14)
        
        save_path = os.path.join(self.args.output_dir, "tsne_embedding.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Statistics] t-SNE plot saved to {save_path}")
        
    # ================= 5. 3D 映射数据 =================
    def save_3d_mapping(self):
        """保存3D结构映射所需的文件"""
        print("[3D Mapping] Generating structure mapping files...")
        
        if not self.attentions:
            print("[Warning] No attention data for 3D mapping")
            return
        
        # 选择高置信度正样本
        high_conf_samples = [
            att for att in self.attentions
            if att['pred'] == 1 and att['prob'] > 0.95 and att['label'] == 1
        ]
        
        if not high_conf_samples:
            print("[Warning] No high-confidence positive samples found")
            return
        
        sample = high_conf_samples[0]
        
        # 保存序列
        seq_path = os.path.join(self.args.output_dir, "structure_target.fasta")
        with open(seq_path, "w") as f:
            f.write(f">{sample['id']}\n{sample['seq']}\n")
        
        # 保存注意力权重
        att = sample['attention'][0, 1:len(sample['seq'])+1]  # 去除CLS和EOS
        att_norm = (att - att.min()) / (att.max() - att.min() + 1e-8) * 100
        
        att_path = os.path.join(self.args.output_dir, "structure_weights.txt")
        np.savetxt(att_path, att_norm, fmt="%.4f")
        
        print(f"[3D Mapping] Sequence saved to {seq_path}")
        print(f"[3D Mapping] Weights saved to {att_path}")
        
    # ================= 主执行流程 =================
    def run(self):
        """根据模式执行相应分析"""
        print(f"\n{'='*60}")
        print(f"Running Analysis Mode: {self.args.mode.upper()}")
        print(f"{'='*60}\n")
        
        # 推理
        self.run_inference()
        
        # 根据模式执行不同任务
        if self.args.mode == 'evaluation':
            self.save_predictions()
            self.compute_metrics()
            
        elif self.args.mode == 'attention':
            self.save_predictions()
            self.compute_metrics()
            self.visualize_attention()
            
        elif self.args.mode == 'tsne':
            self.save_predictions()
            self.plot_probability_histogram()
            self.plot_tsne()
            
        elif self.args.mode == 'full':
            self.save_predictions()
            self.compute_metrics()
            self.plot_probability_histogram()
            self.visualize_attention()
            self.plot_tsne()
            self.save_3d_mapping()
        
        print(f"\n{'='*60}")
        print(f"Analysis completed! Results saved to {self.args.output_dir}")
        print(f"{'='*60}\n")


# ================= 数据集类 =================
class SimpleDataset(Dataset):
    """简单的序列数据集"""
    def __init__(self, sequences, labels, ids):
        self.sequences = sequences
        self.labels = labels
        self.ids = ids
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'seq': self.sequences[idx],
            'label': self.labels[idx],
            'id': self.ids[idx]
        }


# ================= 参数解析 =================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Integrated Model Analysis Tool for ESM-2 LoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ===== 必需参数 =====
    parser.add_argument(
        "-i", "--input_fasta",
        required=True,
        help="Path to input FASTA file"
    )
    parser.add_argument(
        "-m", "--lora_model",
        required=True,
        help="Path to trained LoRA model directory"
    )
    
    # ===== 分析模式 =====
    parser.add_argument(
        "--mode",
        type=str,
        default="evaluation",
        choices=["evaluation", "attention", "tsne", "full"],
        help="""Analysis mode:
        - evaluation: 基础评估 (指标+CSV)
        - attention: 评估+注意力热力图
        - tsne: 评估+概率直方图+t-SNE
        - full: 所有功能"""
    )
    
    # ===== 输出配置 =====
    parser.add_argument(
        "-o", "--output_dir",
        default="./results/integrated_analysis",
        help="Output directory for all results"
    )
    
    # ===== 模型配置 =====
    parser.add_argument(
        "--base_model",
        default="facebook/esm2_t33_650M_UR50D",
        help="Base model name from Hugging Face"
    )
    parser.add_argument(
        "--cache_dir",
        default="./model",
        help="Cache directory for models"
    )
    
    # ===== 推理配置 =====
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    
    # ===== 可视化配置 =====
    parser.add_argument(
        "--num_attention_plots",
        type=int,
        default=3,
        help="Number of attention maps to generate"
    )
    parser.add_argument(
        "--tsne_samples",
        type=int,
        default=3000,
        help="Maximum samples for t-SNE (to save time)"
    )
    
    return parser.parse_args()


# ================= 主程序入口 =================
if __name__ == "__main__":
    args = parse_args()
    
    # 创建分析器
    analyzer = ModelAnalyzer(args)
    
    # 运行分析
    analyzer.run()


"""
使用示例：

# 1. 基础评估模式（最快）
python integrated_analysis.py \
    --mode evaluation \
    --input_fasta test.fasta \
    --lora_model ./results/model/final_lora_model \
    --output_dir ./results/analysis_eval

# 2. 注意力可视化模式
python integrated_analysis.py \
    --mode attention \
    --input_fasta test.fasta \
    --lora_model ./results/model/final_lora_model \
    --output_dir ./results/analysis_attention \
    --num_attention_plots 5

# 3. t-SNE可视化模式
python integrated_analysis.py \
    --mode tsne \
    --input_fasta test.fasta \
    --lora_model ./results/model/final_lora_model \
    --output_dir ./results/analysis_tsne \
    --tsne_samples 5000

# 4. 完整分析模式（所有功能）
python integrated_analysis.py \
    --mode full \
    --input_fasta test.fasta \
    --lora_model ./results/model/final_lora_model \
    --output_dir ./results/analysis_full \
    --batch_size 32

# 5. 自定义配置
python integrated_analysis.py \
    --mode full \
    --input_fasta ../results/data/input/test.fasta \
    --lora_model ../results/model/final_lora_model \
    --output_dir ../results/final_output \
    --base_model facebook/esm2_t33_650M_UR50D \
    --batch_size 32 \
    --max_length 1024 \
    --num_attention_plots 10 \
    --tsne_samples 4000 > ../results/logs/final_output.log 2>&1
"""
