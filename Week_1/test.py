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
#token句子 函数定义，定义 分隔符标记
tokenizer=get_tokenizer("basic_english")