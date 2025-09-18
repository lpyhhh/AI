#/usr/bin/env python
"""
Linux如何声明环境变量
三种：临时 全局 全用户
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
export PATH=/usr/local/bin:$PATH

vim ~/.bashrc

vim /etc/profile
export JAVA=

#? 有问题
"""

#1 加载模型 pipeline
"""
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(generator)
"""
#2模型内部是如何运行的？
#token+model+head_model+pretraining

# myself
#token,model,head_model,
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch

checkpoint="distilbert-base-uncased-finetuned-sst-2-english"
token=AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs=token(raw_inputs,padding=True,truncation=True,return_tensors="pt") #长度填充，位置填充
#print(inputs)

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)
out=model(**inputs)
print(out.logits)

pred=torch.nn.functional.softmax(out.logits,dim=-1)
print(pred)

#model
#模型内部是如何组成的？
from transformers import BertConfig, BertModel
contig=BertConfig()
model=BertModel(contig)#只加载配置模型
model=BertModel.from_pretrained("bert-ase-cased")#经过预训练

model.save_pretrained("../model/bert")

# 标记器