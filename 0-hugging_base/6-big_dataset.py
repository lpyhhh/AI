#/usr/bin/env python
'''
如何在新的文本语料库上训练一个类似于给定 checkpoint 所使用的新 tokenizer
快速 tokenizer 的特殊功能
目前 NLP 中使用的三种主要子词 tokenization 算法之间的差异
如何使用🤗 Tokenizers 库从头开始构建 tokenizer 并在一些数据上进行训练
'''
#1 训练一个tokenzier语料库
#/home/ec2-user/project/lpy/datasets

#1.1 使用python 的语料库
from datasets import load_dataset

raw_datasets = load_dataset(
    "json",
    data_files="/home/ec2-user/project/lpy/datasets/python.jsonl.gz",
    split="train"
)