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
#以下，是根据流程引用包
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import *
from self_attnetion import make_model

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
    seq=nn.utils.rnn.pad_sequence(seq,batch_first=True,padding_value=PAD_IDX)
    return label,seq
#4 封装
train_loader=DataLoader(train_data.with_format("torch"),batch_size=64,shuffle=True,collate_fn=collate)
test_loader=DataLoader(test_data.with_format("torch"),batch_size=64,shuffle=True,collate_fn=collate)
#batch = next(iter(train_loader))
#print(batch)  # 输出 (labels, sequences) 元组

########## 4流程
#数据+模型=一个class
#def make_model(source_vocab,target_vocab,N=6,d_model=512,d_ff=2048,head=8,dropout=0.1):
class TransformerClf(nn.Module):
    def __init__(self,src_vocab,N=6,d_model=512,d_ff=2048,head=8,dropout=0.1,num_classes=2):
        super(TransformerClf,self).__init__()

        base=make_model(src_vocab,src_vocab,d_model=d_model,N=N,d_ff=d_ff,dropout=dropout)

        self.embed=base.src_embed #encoder输出 数据
        self.encoder=base.encoder
        self.classifier=nn.Linear(d_model,num_classes)

    def forward(self,x):
        mask=(x==PAD_IDX).to(x.device) #填充的地方，给上mask标签，不用注意这些地方
        x=self.embed(x) #数据处理完成
        
        out=self.encoder(x,mask.unsqueeze(1).unsqueeze(1))
        mask_exp=mask.unsqueeze(-1)
        out=out.masked_fill(mask_exp,0.) # 删除掩码部分，数字改为0
        sum_=out.sum(dim=1) #长度综合
        cnt=(~mask).sum(dim=1,keepdim=True).float() #每条有效长度
        avg=sum_/cnt #
        return self.classifier(avg)

########## 5训练
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model=TransformerClf(len(vocab)).to(device) #长度确定

opt=torch.optim.Adam(model.parameters(),lr=3e-4) #lr学习率
criterion=nn.CrossEntropyLoss() #loss损失函数

def accuracy(preds,y):#准确率计算，预测结果和标志相同
    return (preds.argmax(1)==y).float().mean().item()

"""
训练流程
for epoch轮次：

    轮次数据给显卡：
        标签和数据给显卡
        初始化并带入模型计算
        结果 损失 初始化

    验证准确性
"""
for epoch in range(3):
    model.train()
    losses,accs=[],[]
    for label,text in train_loader:
        label, text = label.to(device), text.to(device)

        opt.zero_grad() #lr
        logits=model(text)
        loss=criterion(logits,label)
        loss.backward()
        opt.step()

        losses.append(loss.item())
        accs.append(accuracy(logits, label))

        print(f"Epoch {epoch+1}  train loss {sum(losses)/len(losses):.3f}  "
          f"acc {sum(accs)/len(accs):.3f}")

    # 验证集
    model.eval()
    with torch.no_grad():
        test_acc = [accuracy(model(text.to(device)), label.to(device))
                    for label, text in test_loader]
    print(f"         test acc {sum(test_acc)/len(test_acc):.3f}\n")