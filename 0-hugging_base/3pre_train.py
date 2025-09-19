#/usr/bin/env python

###3微调模型
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 和之前一样
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]

batch=tokenizer(sequences,padding=True,truncation=True,return_tensors="pt")
#加上标签
batch["labels"]=torch.tensor([1,1])
#print(batch)
#优化模型的参数
optimizer=AdamW(model.parameters()) #优化器，优化模型所有参数
loss=model(**batch).loss #输入数据与训练
loss.backward() #反向传播算法，自动计算损失函数对所有可训练参数的梯度
optimizer.step() #更新参数


###3.1 数据集加载
#!pip install datasets
from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets=load_dataset("glue","mrpc")
#print(raw_datasets)
#数据集情况：训练集、验证集和测试集；每个集合有四列：句子1 2 标签 分段

#查看数据情况
raw_train=raw_datasets['train']
#print(raw_train)
#print(raw_train.features) #查看标签特征

#预处理数据集，我们需要将文本转换为数字
checkpoint = "bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

def tokenize_funcation(example):
    return tokenizer(example["sentence1"],example["sentence2"],truncation=True)
#tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
#tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
tokenized_datasets=raw_datasets.map(tokenize_funcation,batched=True)
#print(tokenized_datasets)
'''
train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 3668
    })
    填充之后，数据集变为'input_ids', 'token_type_ids', 'attention_mask' 是所需要的矩阵数据
'''
#动态填充
from transformers import DataCollatorWithPadding
data_collator=DataCollatorWithPadding(tokenizer=tokenizer) #填充的是模型
sample=tokenized_datasets['train'][:8]
sample={k:v for k,v in sample.items() if k not in ["idx", "sentence1", "sentence2"]}
#删除不需要的列 操作方式
[len(x) for x in sample['input_ids']]

###3.2 训练
#!pip install transformers[torch]
#超参数，模型，训练数据加载，开始微调
from transformers import TrainingArguments,AutoModelForSequenceClassification
training_args=TrainingArguments("test-trainer")