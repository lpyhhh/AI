#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Extraction (FE)
重新生成 嵌入向量，也就是预处理咯

数据导入，模型导入，数据与模型处理，测试运行
"""
import time 
import h5py
import argparse

import torch
from transformers import T5EncoderModel, T5Tokenizer

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
print("using device ",device)

# ########## 数据处理
def read_fasta(fasta_path):
    """
    读取 FASTA 文件，处理id和seq，返回一个 dict：{uniprot_id: sequence}。
    header 行以 '>' 开始，取整行作为 ID，但将 '/' 和 '.' 替换为 '_'（因为 HDF5 dataset 名中这些字符可能有问题）。
    读取序列行时去掉所有空白并转为大写，移除 '-'（gap）。
    注意：如果同一个 header 出现多行序列，会拼接成一条

    fasta_path : input file
    """
    sequences = dict()
    with open(fasta_path,'r') as fasta:
        for line in fasta:
            if line.startswith('>'):
                uniprot_id = line.replace('>','').split()[0].strip()
                uniprot_id = uniprot_id.replace('/','').replace('.','')
                sequences[uniprot_id] = ''
            else:
                sequences[uniprot_id] += ''.join(line.split()).upper().replace('-','')
        #print(sequences)
    return sequences

#read_fasta("/home/ec2-user/project/test.fa")

# ########## 模型构建
"""
加载模型，输出模型目录，并缓存到指定目录，如果 device 是 CPU 则把模型 cast 到 float32，返回 model（eval 模式）和 vocab（tokenizer）
"""
def get_T5_model(model_dir,transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"):
    """
    model_dir : model output dir
    transformer_link : model location of huggingface
    """
    if model_dir is not None:
        print("##########################")
        print("pretrain model location:{}".format(model_dir))
        print("##########################")
    
    model = T5EncoderModel.from_pretrained(transformer_link,cache_dir=model_dir)
    if device == torch.device("cpu"):
        model.to(torch.float32)
    model = model.to(device).half()
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(transformer_link,do_lower_case=False)
    return model,vocab
#get_T5_model(model_dir="/home/ec2-user/project/model")

# ########## 数据与模型处理
"""
块处理代码：
1 加载类，并验证加载情况
2 统计字典序列基础信息：平均长度、超长序列数量 []列表推导式
3 构建批量运行
"""
def get_embeddings(
    seq_path,
    model_dir,
    emb_path,
    per_protein,# output vector 是否池化：整个蛋白一个向量或者多个向量
    max_residues=4000, #单批次 残基
    max_seq_len=4000,
    max_batch=100
):
    seq_dict = dict() # 读取序列
    emb_path = dict()#最终的嵌入，用作调试
    seq_dict = read_fasta(seq_path)
    model,vocab = get_T5_model(model_dir)
    # test of output
    print('########################################')
    print("example sequence >{}\n{}".format(# dict{} 测试输出，首先要知道数据情况 选择处理方式
    next(iter(seq_dict.keys())),
    next(iter(seq_dict.values()))
    ))

    # 2
    avg_length = sum([len(seq) for _,seq in seq_dict.items()]) / len(seq_dict)
    #字典长度=keys长度，
    #print(len(seq_dict))
    n_long = sum([1 for _,seq in seq_dict.items() if len(seq)>max_seq_len])
    sorted(seq_dict,key=lambda kv:len(kv[1]),reverse=True)
    #test calculate results
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences {}: {}".format(max_seq_len, n_long))

    # 3 batch 序列处理，
    for seq_idx,(pdb_id,seq) in enumerate(seq_dict.items(),1):#处理每条序列
        seq = seq.replace('U','X').replace('Z','X').replace('O','X')
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))
        n_res_batch = sum([s_len for _,_,seq_len in batch]) + seq_len
    
        if len(batch)>=max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids,seqs,seq_lens = zip(*batch)
            batch = list()# 列表化

            token_encoding = vocab.batch_encode_plus(seqs,add_apecial_tokens=True,padding="longest")
            input_ids = torch_tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)


get_embeddings(seq_path="/home/ec2-user/project/test.fa",emb_path="/home/ec2-user/project/emb",model_dir="/home/ec2-user/project/model",per_protein=False)