#!/usr/bin/env python3
"""
input:seqences file
output:over trained LoRA model + training curve plot curve曲线 plot图

requirements:
master AI coding architecture
master coding for class encapsulation 封装
"""
import argparse
import os
from Bio import SeqIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from datasets import Dataset
from transformers import ( #padding,token,class,train class,train
    DataCollatorWithPadding,
    EsmTokenizer,
    EsmForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef #model result analysis
from peft import LoraConfig, get_peft_model, TaskType

class BioTrainer:
    def __init__(self,args):
        """
        initialize the positions of data and the models
        """
        self.args = args
        self._setup_directories()

        self.train_dataset = None 
        self.val_dataset = None
        self.model = None

        self.tokenizer = EsmTokenizer.from_pretrained(args.model_name)

    def _setup_directories(self):
        """private method: to create the required output directories"""
        os.makedirs(self.args.output_dir,exist_ok=True)
        os.makedirs(self.args.log_dir,exist_ok=True)

    # ================= 1. 数据处理 =================
    #主要功能实现：数据划分：读取数据，解析pos nog，token化数据
    def load_datasets(self):
        #读取数据，解析pos nog，
        print("[Data] Loading datasets...")
        self.train_dataset = self._parse_fasta(self.args.train_path)
        self.val_dataset = self._parse_fasta(self.args.val_path)

        self._tokenizer_datasets()

    def _parse_fasta(self, file_path):
        """文件存在，用列表对序列label判定并分类"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        seqences = []
        labels = []
        for record in SeqIO.parse(file_path, "fasta"):
            seqences.append(str(record.seq))
            if 'POS' in record.id:
                labels.append(1)
            elif 'NEG' in record.id:
                labels.append(0)
            else:
                labels.append(0)
        return Dataset.from_dict({'sequence':seqences,'labels':labels})

    # ================= 3. 微调数据 (Tokenization) =================   
    def _tokenizer_datasets(self):
        """内部方法：对数据进行分词和格式化"""
        print("[Data] Tokenizing...")
        def tokenize_function(examples):
            return self.tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=self.args.max_length
            )
        
        self.train_dataset = self.train_dataset.map(tokenize_function,batched=True)
        self.val_dataset = self.val_dataset.map(tokenize_function,batched=True)

        self._format_dataset(self.train_dataset)
        self._format_dataset(self.val_dataset)

    def _format_dataset(self, dataset):
        """重命名列并设置 Tensor 格式"""
        if "label" in dataset.column_names:
            dataset = dataset.rename_column("label","labels")
        dataset.set_format("torch",columns=["input_ids", "attention_mask", "labels"])
        return dataset

    # ================= 2. 模型导入与 LoRA 配置 =================
    def setup_model(self):
        """模型加载与 LoRA 配置"""
        #模型
        model = EsmForSequenceClassification.from_pretrained(
            self.args.model_name,
            num_labels=2, #二分类
            cache_dir=self.args.cache_dir
        )

        # LoRA 
        lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            target_modules=["query", "key", "value"], # ESM 常用 target
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )

        # 导入模型
        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()

    # ================= 4. 训练与评估 =================
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
            "mcc": matthews_corrcoef(labels, predictions)
        } 
    
    def train(self):
        """训练模型"""
        if self.model is None:
            raise ValueError("Model is not set up. Please call setup_model() first.")
        
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            logging_dir=self.args.log_dir,
            report_to="tensorboard",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            learning_rate=self.args.learning_rate,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.grad_accum,
            per_device_eval_batch_size=self.args.batch_size,
            fp16=True, # 开启混合精度
            num_train_epochs=self.args.epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics,
        )

        last_checkpoint = self._find_last_checkpoint()

        print("\n" + "="*30)
        print("Starting Training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)

        # 保存
        final_path = os.path.join(self.args.output_dir, "final_lora_model")
        trainer.save_model(final_path)
        print(f"Model saved to {final_path}")

        # 绘图
        self._plot_history(trainer.state.log_history)
    
    def _find_last_checkpoint(self):
        """检测是否存在断点"""
        if os.path.isdir(self.args.output_dir):
            checkpoints = [d for d in os.listdir(self.args.output_dir) if d.startswith("checkpoint")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                ckpt_path = os.path.join(self.args.output_dir, checkpoints[-1])
                print(f"Resuming from checkpoint: {ckpt_path}")
                return ckpt_path
        return None

    def _plot_history(self, log_history):
        """绘制训练曲线 (包含之前的清洗逻辑)"""
        print("Plotting training curves...")
        df = pd.DataFrame(log_history)
        
        # 清洗逻辑：移除重跑前的旧日志
        if len(df) > 1 and 'epoch' in df.columns:
            restart_points = df[df['epoch'].diff() < 0].index.tolist()
            if restart_points:
                df = df.iloc[restart_points[-1]:].reset_index(drop=True)

        train_logs = df[df['loss'].notna() & df['eval_loss'].isna()]
        eval_logs = df[df['eval_loss'].notna()]
        
        if len(train_logs) == 0: 
            return

        plt.figure(figsize=(12, 5))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(train_logs['epoch'], train_logs['loss'], label='Train Loss')
        if not eval_logs.empty:
            plt.plot(eval_logs['epoch'], eval_logs['eval_loss'], label='Val Loss')
        plt.title('Loss')
        plt.legend()
        
        # Metrics
        plt.subplot(1, 2, 2)
        if not eval_logs.empty and 'eval_f1' in eval_logs:
            plt.plot(eval_logs['epoch'], eval_logs['eval_f1'], label='F1')
            plt.plot(eval_logs['epoch'], eval_logs['eval_mcc'], label='MCC')
            plt.title('Metrics')
            plt.legend()
            
        save_path = os.path.join(self.args.output_dir, "training_curves.png")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Curves saved to {save_path}")

# ================= 0. 程序入口与配置 =================
def parse_args():
    """
    Parameters 参数
    """
    parser = argparse.ArgumentParser(description="ESM-2 LoRA Fine-tuning")

    parser.add_argument("--train_path", type=str, required=True, help="path to training FASTA")
    parser.add_argument("--val_path", type=str, required=True, help="path to validation FASTA")
    parser.add_argument("--output_dir", type=str, default="./results/model", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="./results/logs", help="Log directory")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory")

    # 模型参数
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    
    # LoRA 参数
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trainer = BioTrainer(args)
    trainer.load_datasets()
    trainer.setup_model()
    trainer.train()

"""
python3 test.py \
--train_path ../results/data/input/train.fasta \
--val_path ../results/data/input/val.fasta \
--output_dir ../results/model \
--log_dir ../results/logs \
--model_name facebook/esm2_t33_650M_UR50D \
--max_length 1024 \
--batch_size 8 \
--grad_accum 1 \
--epochs 6 \
--learning_rate 5e-5 \
--lora_rank 8 \
--lora_alpha 32 \
--lora_dropout 0.05 > ../results/logs/lora_training.log 2>&1
"""
#required 和 default冲突

