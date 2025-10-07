#!/usr/bin/env python
import collections
import os
import random
import torch
import math
import copy
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import device
from torch.autograd import Variable
#一下是根据流程引用包
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import *

#这些是超参数，一般是用于一个文件保存
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH   = 128
MAXLEN  = 512
D_MODEL = 128
NHEAD   = 4
NUM_CLS = 2
EPOCHS  = 3


########## 数据
imdb=load_dataset("imdb")
train_data=imdb['train']
test_data=imdb['test']
#print(train_data) #text label 25k

########## 清洗

########## 构造输入
#token句子。转化为 数字索引
tokenizer=get_tokenizer("basic_english")
#1 句子token化，训练集。 构建分词化，构建句子开始和结束标志
def yield_tokens():
    for sample in train_data:
        yield tokenizer(sample["text"])

vocab=build_vocab_from_iterator(yield_tokens(),specials=["<pad>", "<unk>"],min_freq=2)
vocab.set_default_index(vocab["<unk>"])
PAD_IDX=vocab["<pad>"]
#2 
def encode_text(text):
    return vocab(tokenizer(text))[:512]
#3 标签和内容转化
def collate(batch):
    label=torch.tensor([int(b["label"]) for b in batch])
    seq=[torch.tensor(encode_text(b["text"])) for b in batch]
    