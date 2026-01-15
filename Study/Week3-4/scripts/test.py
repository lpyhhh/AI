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
import Bio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from dataset import Dataset
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
        self.model = model

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
        lables = []
        for record in SeqIO.parse(file_path, "fasta"):
            seqences.append(str(record.seq))
            if 'POS' in record.id:
                lables.append(1)
            elif 'NEG' in record.id:
                lables.append(0)
            else:
                lables.append(0)
        return Dataset.from_dict({'sequence':seqences,'lable':lables})

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
        if "label" in dataset.column.names:
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

        # 导入模型

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

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trainer = BioTrainer(args)
    trainer.load_datasets()

"""
parser写法
add_argment内容
"""
#required 和 default冲突

