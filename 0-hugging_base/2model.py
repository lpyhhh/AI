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
#print(out.logits)

pred=torch.nn.functional.softmax(out.logits,dim=-1)
#print(pred)

#model
#模型内部是如何组成的？
from transformers import BertConfig, BertModel
contig=BertConfig()
#model=BertModel(contig)#只加载配置模型
model=BertModel.from_pretrained("bert-ase-cased")#经过预训练

model.save_pretrained("../model/bert")

###2.4 标记器
'''
目的：输入序列如何转化为矩阵
model input 是矩阵
token 把输入数据 转化为 矩阵
三种：基于 单词 字母 子词subword
'''
#加载tokenizer并保存到本地
from transformers import BertTokenizer
token=BertTokenizer.from_pretrained("bert-base-cased")
token("Using a Transformer network is simple",",ni shi ge da sha bi")
token.save_pretrained("./model_test")

#编码
#序列-子词
from transformers import AutoTokenizer
token=AutoTokenizer.from_pretrained("bert-base-cased")
seq="Using a Transformer network is simple",",ni shi ge da sha bi"
tokens=token.tokenize(seq) #
#print("序列-子词")
#print(tokens)
ids=token.convert_tokens_to_ids(tokens) #
#print("子词-矩阵")
#print(ids)
decode=token.decode(ids) #
#print("解码")
#print(decode)


###课后作业
#1 浅析模型的基本使用流程 从token model head_model 
"""
基本使用流程
输入句子sequence=
把句子转化为矩阵 auto（）
使用模型把矩阵后的序列进行预测 model（）
加上模型头 对句子类型 进行分类
"""
#模型基本使用
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

token=tokenizer(sequence,padding=True,truncation=True,return_tensors="pt")
pre=model(**token)
#print(pre)

#加token后
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sequence = "I've been waiting for a HuggingFace course my whole life."


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
token=AutoTokenizer.from_pretrained(checkpoint)
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)

tokens=token.tokenize(sequence)
ids=token.convert_tokens_to_ids(tokens)
#这里面保存的是单个句子的矩阵化，而不是 model所需要的信息，三维矩阵

