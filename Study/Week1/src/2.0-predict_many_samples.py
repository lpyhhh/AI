#!/usr/bin/env python
# encoding: utf-8
'''
这个文件的作用是
从输入的FASTA文件中读取蛋白质序列数据，使用预训练的深度学习模型对这些序列进行预测，并将预测结果保存到指定的输出文件中。
它支持多标签分类、多分类和二分类任务，能够加载模型、分词器和相关配置，处理输入数据并生成预测结果

*Copyright (c) 2023, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.

@author: Hey  # 作者: Hey
@email: sanyuan.**@**.com  # 邮箱: sanyuan.**@**.com
@tel: 137****6540  # 电话: 137****6540
@datetime: 2023/4/10 18:26  # 日期时间: 2023/4/10 18:26
@project: DeepProtFunc  # 项目: DeepProtFunc
@file: predict_many_samples  # 文件: predict_many_samples
@desc: predict many samples from file  # 描述: 从文件中预测多个样本
'''
import argparse  # 导入argparse模块，用于解析命令行参数
import csv  # 导入csv模块，用于处理CSV文件
import numpy as np  # 导入numpy模块，并简写为np，用于科学计算
import os, sys, json, codecs  # 导入os、sys、json和codecs模块
from subword_nmt.apply_bpe import BPE  # 从subword_nmt.apply_bpe模块导入BPE类
from transformers.models.bert.configuration_bert import BertConfig  # 从transformers.models.bert.configuration_bert模块导入BertConfig类
from transformers.models.bert.tokenization_bert import BertTokenizer  # 从transformers.models.bert.tokenization_bert模块导入BertTokenizer类
sys.path.append(".")  # 将当前目录添加到系统路径
sys.path.append("..")  # 将上一级目录添加到系统路径
sys.path.append("../src")  # 将src目录添加到系统路径
try:
    from common.multi_label_metrics import *  # 从common.multi_label_metrics模块导入所有内容
    from protein_structure.predict_structure import predict_embedding, predict_pdb, calc_distance_maps  # 从protein_structure.predict_structure模块导入多个函数
    from utils import set_seed, plot_bins, csv_reade, fasta_reader, clean_seq  # 从utils模块导入多个函数
    from SSFN.model import *  # 从SSFN.model模块导入所有内容
    from data_loader import load_and_cache_examples, convert_examples_to_features, InputExample, InputFeatures  # 从data_loader模块导入多个函数和类
except ImportError:
    from src.common.multi_label_metrics import *  # 如果导入失败，从src.common.multi_label_metrics模块导入所有内容
    from src.protein_structure.predict_structure import predict_embedding, predict_pdb, calc_distance_maps  # 从src.protein_structure.predict_structure模块导入多个函数
    from src.utils import set_seed, plot_bins, csv_reader, fasta_reader, clean_seq  # 从src.utils模块导入多个函数
    from src.SSFN.model import *  # 从src.SSFN.model模块导入所有内容
    from src.data_loader import load_and_cache_examples, convert_examples_to_features, InputExample, InputFeatures  # 从src.data_loader模块导入多个函数和类

import logging  # 导入logging模块，用于日志记录
logger = logging.getLogger(__name__)  # 创建一个日志记录器

def llprint(message):  # 定义llprint函数，用于打印消息
    sys.stdout.write(message + "\n")  # 将消息写入标准输出
    sys.stdout.flush()  # 刷新输出缓冲区

def load_label_code_2_name(args, filename):  # 定义load_label_code_2_name函数，用于加载标签代码与名称的映射
    '''
    load the mapping between the label name and label code
    加载标签名称与标签代码之间的映射
    :param args: 参数
    :param filename: 文件名
    :return: 返回标签代码与名称的映射
    '''
    label_code_2_name = {}  # 初始化一个空字典
    label_filepath = "../dataset/%s/%s/%s/%s" % (args.dataset_name, args.dataset_type, args.task_type, filename)  # 构造标签文件路径
    if label_filepath and os.path.exists(label_filepath):  # 如果标签文件路径存在
        with open(label_filepath, "r") as rfp:  # 打开文件
            for line in rfp:  # 遍历文件中的每一行
                strs = line.strip().split("###")  # 按###分割每一行
                label_code_2_name[strs[0]] = strs[1]  # 将分割后的内容存入字典
    return label_code_2_name  # 返回标签代码与名称的映射

def load_args(log_dir):  # 定义load_args函数，用于加载模型运行参数
    '''
    load model running args
    加载模型运行参数
    :param log_dir: 日志目录
    :return: 返回配置对象
    '''
    print("-" * 25 + "log dir:" + "-" * 25)  # 打印日志目录
    print(log_dir)  # 打印日志目录
    print("-" * 60)  # 打印分隔线
    log_filepath = os.path.join(log_dir, "logs.txt")  # 构造日志文件路径
    if not os.path.exists(log_filepath):  # 如果日志文件不存在
        raise Exception("%s not exists" % log_filepath)  # 抛出异常
    with open(log_filepath, "r") as rfp:  # 打开日志文件
        for line in rfp:  # 遍历文件中的每一行
            if line.startswith("{"):  # 如果行以"{"开头
                obj = json.loads(line.strip())  # 解析JSON对象
                return obj  # 返回对象
    return {}  # 返回空字典

def load_model(args, model_dir):  # 定义load_model函数，用于加载模型
    '''
    load the model
    加载模型
    :param args: 参数
    :param model_dir: 模型目录
    :return: 返回配置、子词、序列分词器、结构分词器、模型、标签ID到名称的映射、标签名称到ID的映射
    '''
    # load tokenizer and model
    # 加载分词器和模型
    device = torch.device(args.device)  # 获取设备
    config_class, model_class, tokenizer_class = BertConfig, SequenceAndStructureFusionNetwork, BertTokenizer  # 获取配置类、模型类和分词器类

    # config = config_class(**json.load(open(os.path.join(model_dir, "config.json"), "r"), encoding="UTF-8"))
    config = config_class(**json.load(open(os.path.join(model_dir, "config.json"), "r")))  # 加载配置
    # for sequence
    # 对于序列
    subword = None  # 初始化子词为None
    if args.has_seq_encoder:  # 如果有序列编码器
        seq_tokenizer = tokenizer_class.from_pretrained(
            os.path.join(model_dir, "sequence"),
            do_lower_case=args.do_lower_case
        )  # 加载序列分词器
        if args.subword:  # 如果有子词
            bpe_codes_prot = codecs.open(args.codes_file)  # 打开子词代码文件
            subword = BPE(bpe_codes_prot, merges=-1, separator='')  # 创建BPE对象
    else:
        seq_tokenizer = None  # 否则，序列分词器为None

    if args.has_struct_encoder:  # 如果有结构编码器
        struct_tokenizer = tokenizer_class.from_pretrained(
            os.path.join(model_dir, "struct"),
            do_lower_case=args.do_lower_case
        )  # 加载结构分词器
    else:
        struct_tokenizer = None  # 否则，结构分词器为None

    model = model_class.from_pretrained(model_dir, args=args)  # 加载模型

    model.to(device)  # 将模型移动到设备
    model.eval()  # 设置模型为评估模式

    # load labels
    # 加载标签
    label_filepath = args.label_filepath  # 获取标签文件路径
    label_id_2_name = {}  # 初始化标签ID到名称的映射
    label_name_2_id = {}  # 初始化标签名称到ID的映射
    with open(label_filepath, "r") as fp:  # 打开标签文件
        for line in fp:  # 遍历文件中的每一行
            if line.strip() == "label":  # 如果行内容为"label"
                continue  # 跳过
            label_name = line.strip()  # 获取标签名称
            label_id_2_name[len(label_id_2_name)] = label_name  # 将标签ID到名称的映射存入字典
            label_name_2_id[label_name] = len(label_name_2_id)  # 将标签名称到ID的映射存入字典

    print("-" * 25 + "label_id_2_name:" + "-" * 25)  # 打印标签ID到名称的映射
    if len(label_id_2_name) < 20:  # 如果标签数量小于20
        print(label_id_2_name)  # 打印标签ID到名称的映射
    print("label size: ", len(label_id_2_name))  # 打印标签数量
    print("-" * 60)  # 打印分隔线

    return config, subword, seq_tokenizer, struct_tokenizer, model, label_id_2_name, label_name_2_id  # 返回配置、子词、序列分词器、结构分词器、模型、标签ID到名称的映射、标签名称到ID的映射

def transform_sample_2_feature(
        args,
        row,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True
):  # 定义transform_sample_2_feature函数，用于将样本转换为特征
    '''
    batch sample transform to batch input
    批量样本转换为批量输入
    :param args: 参数
    :param row: [protein_id, seq] 样本行
    :param seq_tokenizer: 序列分词器
    :param subword: 子词
    :param struct_tokenizer: 结构分词器
    :param pad_on_left: 是否在左侧填充
    :param pad_token: 填充标记
    :param pad_token_segment_id: 填充标记段ID
    :param mask_padding_with_zero: 是否用零填充掩码
    :return: 返回批量信息和批量输入
    '''
    features = []  # 初始化特征列表
    batch_info = []  # 初始化批量信息列表
    # id, seq
    # ID和序列
    prot_id, protein_seq = row[0], row[1]  # 获取蛋白质ID和序列
    batch_info.append(row)  # 将样本行添加到批量信息列表
    assert seq_tokenizer is not None or struct_tokenizer is not None or args.embedding_type is not None  # 断言序列分词器、结构分词器或嵌入类型不为空
    if seq_tokenizer:  # 如果有序列分词器
        if subword:  # 如果有子词
            seq_to_list = subword.process_line(protein_seq).split(" ")  # 处理序列并分割为列表
        else:
            seq_to_list = [v for v in protein_seq]  # 将序列转换为列表
        cur_seq_len = len(seq_to_list)  # 获取当前序列长度
        if cur_seq_len > args.seq_max_length - 2:  # 如果当前序列长度大于最大长度减2
            if args.trunc_type == "left":  # 如果截断类型为左侧
                seq_to_list = seq_to_list[2 - args.seq_max_length:]  # 从左侧截断序列
            else:
                seq_to_list = seq_to_list[:args.seq_max_length - 2]  # 从右侧截断序列
        seq = " ".join(seq_to_list)  # 将序列列表转换为字符串
        inputs = seq_tokenizer.encode_plus(
            seq,
            None,
            add_special_tokens=True,
            max_length=args.seq_max_length,
            truncation=True
        )  # 编码序列
        # input_ids: token index list
        # token_type_ids: token type index list
        # input_ids: 标记索引列表
        # token_type_ids: 标记类型索引列表
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]  # 获取输入ID和标记类型ID
        real_token_len = len(input_ids)  # 获取真实标记长度

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # 掩码对真实标记为1，对填充标记为0。只有真实标记会被关注。
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)  # 创建注意力掩码

        # Zero-pad up to the sequence length.
        # 零填充到序列长度
        padding_length = args.seq_max_length - len(input_ids)  # 计算填充长度
        attention_mask_padding_length = padding_length  # 设置注意力掩码填充长度

        if pad_on_left:  # 如果在左侧填充
            input_ids = ([pad_token] * padding_length) + input_ids  # 在左侧填充输入ID
            attention_mask = ([0 if mask_padding_with_zero else 1] * attention_mask_padding_length) + attention_mask  # 在左侧填充注意力掩码
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids  # 在左侧填充标记类型ID
        else:
            input_ids = input_ids + ([pad_token] * padding_length)  # 在右侧填充输入ID
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * attention_mask_padding_length)  # 在右侧填充注意力掩码
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)  # 在右侧填充标记类型ID

        assert len(input_ids) == args.seq_max_length, "Error with input length {} vs {}".format(len(input_ids), args.seq_max_length)  # 断言输入ID长度等于最大序列长度
        assert len(attention_mask) == args.seq_max_length, "Error with input length {} vs {}".format(len(attention_mask), args.seq_max_length)  # 断言注意力掩码长度等于最大序列长度
        assert len(token_type_ids) == args.seq_max_length, "Error with input length {} vs {}".format(len(token_type_ids), args.seq_max_length)  # 断言标记类型ID长度等于最大序列长度
    else:
        input_ids = None  # 否则，输入ID为None
        attention_mask = None  # 注意力掩码为None
        token_type_ids = None  # 标记类型ID为None
        real_token_len = None  # 真实标记长度为None
    if struct_tokenizer:  # 如果有结构分词器
        # for structure
        # 对于结构
        cur_seq_len = len(protein_seq)  # 获取当前序列长度
        seq_list = [ch for ch in protein_seq]  # 将序列转换为列表
        if cur_seq_len > args.struct_max_length:  # 如果当前序列长度大于最大结构长度
            if args.trunc_type == "left":  # 如果截断类型为左侧
                seq_list = seq_list[-args.struct_max_length:]  # 从左侧截断序列
            else:
                seq_list = seq_list[:args.struct_max_length]  # 从右侧截断序列
        seq = " ".join(seq_list)  # 将序列列表转换为字符串
        inputs = struct_tokenizer.encode_plus(
            seq,
            None,
            add_special_tokens=False,
            max_length=args.struct_max_length,
            truncation=True,
            return_token_type_ids=False,
        )  # 编码序列
        struct_input_ids = inputs["input_ids"]  # 获取结构输入ID
        real_struct_node_size = len(struct_input_ids)  # 获取真实结构节点大小
        padding_length = args.struct_max_length - real_struct_node_size if real_struct_node_size < args.struct_max_length else 0  # 计算填充长度
        pdb, mean_plddt, ptm, processed_seq = predict_pdb(
            [prot_id, protein_seq], args.trunc_type,
            num_recycles=4,
            truncation_seq_length=args.truncation_seq_length,
            chunk_size=64,
            cpu_type="cpu-offload"
        )  # 预测PDB
        # if the savepath not exists, create it
        # 如果保存路径不存在，创建它
        if args.pdb_dir:  # 如果有PDB目录
            if not os.path.exists(args.pdb_dir):  # 如果PDB目录不存在
                os.makedirs(args.pdb_dir)  # 创建PDB目录
            pdb_filepath = os.path.join(args.pdb_dir, prot_id.replace("/", "_") + ".pdb")  # 构造PDB文件路径
            with open(pdb_filepath, "w") as wfp:  # 打开PDB文件
                wfp.write(pdb)  # 写入PDB内容
        c_alpha, c_beta = calc_distance_maps(pdb, args.chain, processed_seq)  # 计算距离图
        cmap = c_alpha[args.chain]['contact-map'] if args.cmap_type == "C_alpha" else c_beta[args.chain]['contact-map']  # 获取接触图
        # use the specific threshold to transform the float contact map into 0-1 contact map
        # 使用特定阈值将浮点接触图转换为0-1接触图
        cmap = np.less_equal(cmap, args.cmap_thresh).astype(np.int32)  # 转换接触图
        struct_contact_map = cmap  # 设置结构接触图
        real_shape = struct_contact_map.shape  # 获取真实形状
        if real_shape[0] > args.struct_max_length:  # 如果真实形状大于最大结构长度
            if args.trunc_type == "left":  # 如果截断类型为左侧
                struct_contact_map = struct_contact_map[-args.struct_max_length:, -args.struct_max_length:]  # 从左侧截断接触图
            else:
                struct_contact_map = struct_contact_map[:args.struct_max_length, :args.struct_max_length]  # 从右侧截断接触图
            contact_map_padding_length = 0  # 设置接触图填充长度为0
        else:
            contact_map_padding_length = args.struct_max_length - real_shape[0]  # 计算接触图填充长度
        assert contact_map_padding_length == padding_length  # 断言接触图填充长度等于填充长度

        if contact_map_padding_length > 0:  # 如果接触图填充长度大于0
            if pad_on_left:  # 如果在左侧填充
                struct_input_ids = [pad_token] * padding_length + struct_input_ids  # 在左侧填充结构输入ID
                struct_contact_map = np.pad(struct_contact_map, [(contact_map_padding_length, 0), (contact_map_padding_length, 0)], mode='constant', constant_values=pad_token)  # 在左侧填充结构接触图
            else:
                struct_input_ids = struct_input_ids + ([pad_token] * padding_length)  # 在右侧填充结构输入ID
                struct_contact_map = np.pad(struct_contact_map, [(0, contact_map_padding_length), (0, contact_map_padding_length)], mode='constant', constant_values=pad_token)  # 在右侧填充结构接触图

        assert len(struct_input_ids) == args.struct_max_length, "Error with input length {} vs {}".format(len(struct_input_ids), args.struct_max_length)  # 断言结构输入ID长度等于最大结构长度
        assert struct_contact_map.shape[0] == args.struct_max_length, "Error with input length {}x{} vs {}x{}".format(struct_contact_map.shape[0], struct_contact_map.shape[1], args.struct_max_length, args.struct_max_length)  # 断言结构接触图形状等于最大结构长度
    else:
        struct_input_ids = None  # 否则，结构输入ID为None
        struct_contact_map = None  # 结构接触图为None
        real_struct_node_size = None  # 真实结构节点大小为None

    if args.embedding_type:  # 如果有嵌入类型
        # for embedding
        # 对于嵌入
        embedding_info, processed_seq = predict_embedding(
            [prot_id, protein_seq],
            args.trunc_type,
            "representations" if args.embedding_type == "matrix" else args.embedding_type,
            repr_layers=[-1],
            truncation_seq_length=args.truncation_seq_length - 2,
            device=args.device
        )  # 预测嵌入
        # failure on GPU, then using CPU for embedding
        # 如果在GPU上失败，则使用CPU进行嵌入
        if embedding_info is None:  # 如果嵌入信息为空
            # 失败,则调用cpu进行embedding推理
            # 失败，则调用CPU进行嵌入推理
            embedding_info, processed_seq = predict_embedding(
                [prot_id, protein_seq],
                args.trunc_type,
                "representations" if args.embedding_type == "matrix" else args.embedding_type,
                repr_layers=[-1],
                truncation_seq_length=args.truncation_seq_length - 2,
                device=torch.device("cpu")
            )  # 预测嵌入
        if args.emb_dir:  # 如果有嵌入目录
            if not os.path.exists(args.emb_dir):  # 如果嵌入目录不存在
                os.makedirs(args.emb_dir)  # 创建嵌入目录

            embedding_filepath = os.path.join(args.emb_dir, prot_id.replace("/", "_") + ".pt")  # 构造嵌入文件路径
            torch.save(embedding_info, embedding_filepath)  # 保存嵌入信息
        if args.embedding_type == "contacts":  # 如果嵌入类型为接触
            emb_l = embedding_info.shape[0]  # 获取嵌入长度
            embedding_attention_mask = [1 if mask_padding_with_zero else 0] * emb_l  # 创建嵌入注意力掩码
            if emb_l > args.embedding_max_length:  # 如果嵌入长度大于最大嵌入长度
                if args.trunc_type == "left":  # 如果截断类型为左侧
                    embedding_info = embedding_info[-args.embedding_max_length:, -args.embedding_max_length:]  # 从左侧截断嵌入信息
                else:
                    embedding_info = embedding_info[:args.embedding_max_length, :args.embedding_max_length]  # 从右侧截断嵌入信息
                embedding_attention_mask = [1 if mask_padding_with_zero else 0] * args.embedding_max_length  # 创建嵌入注意力掩码
            else:
                embedding_padding_length = args.embedding_max_length - emb_l  # 计算嵌入填充长度
                if embedding_padding_length > 0:  # 如果嵌入填充长度大于0
                    if pad_on_left:  # 如果在左侧填充
                        embedding_attention_mask = [0 if mask_padding_with_zero else 1] * embedding_padding_length + embedding_attention_mask  # 在左侧填充嵌入注意力掩码
                        embedding_info = np.pad(embedding_info, [(embedding_padding_length, 0), (embedding_padding_length, 0)], mode='constant', constant_values=pad_token)  # 在左侧填充嵌入信息
                    else:
                        embedding_attention_mask = embedding_attention_mask + [0 if mask_padding_with_zero else 1] * embedding_padding_length  # 在右侧填充嵌入注意力掩码
                        embedding_info = np.pad(embedding_info, [(0, embedding_padding_length), (0, embedding_padding_length)], mode='constant', constant_values=pad_token)  # 在右侧填充嵌入信息
        elif args.embedding_type == "matrix":  # 如果嵌入类型为矩阵
            emb_l = embedding_info.shape[0]  # 获取嵌入长度
            embedding_attention_mask = [1 if mask_padding_with_zero else 0] * emb_l  # 创建嵌入注意力掩码
            if emb_l > args.embedding_max_length:  # 如果嵌入长度大于最大嵌入长度
                if args.trunc_type == "left":  # 如果截断类型为左侧
                    embedding_info = embedding_info[-args.embedding_max_length:, :]  # 从左侧截断嵌入信息
                else:
                    embedding_info = embedding_info[:args.embedding_max_length, :]  # 从右侧截断嵌入信息
                embedding_attention_mask = [1 if mask_padding_with_zero else 0] * args.embedding_max_length  # 创建嵌入注意力掩码
            else:
                embedding_padding_length = args.embedding_max_length - emb_l  # 计算嵌入填充长度
                if embedding_padding_length > 0:  # 如果嵌入填充长度大于0
                    if pad_on_left:  # 如果在左侧填充
                        embedding_attention_mask = [0 if mask_padding_with_zero else 1] * embedding_padding_length + embedding_attention_mask  # 在左侧填充嵌入注意力掩码
                        embedding_info = np.pad(embedding_info, [(embedding_padding_length, 0), (0, 0)], mode='constant', constant_values=pad_token)  # 在左侧填充嵌入信息
                    else:
                        embedding_attention_mask = embedding_attention_mask + [0 if mask_padding_with_zero else 1] * embedding_padding_length  # 在右侧填充嵌入注意力掩码
                        embedding_info = np.pad(embedding_info, [(0, embedding_padding_length), (0, 0)], mode='constant', constant_values=pad_token)  # 在右侧填充嵌入信息
        elif args.embedding_type == "bos":  # 如果嵌入类型为BOS
            embedding_attention_mask = None  # 嵌入注意力掩码为None
        else:
            raise Exception("Not support arg: --embedding_type=%s" % args.embedding_type)  # 抛出异常
    else:
        embedding_info = None  # 否则，嵌入信息为None
        embedding_attention_mask = None  # 嵌入注意力掩码为None
    features.append(
        InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            real_token_len=real_token_len,
            struct_input_ids=struct_input_ids,
            struct_contact_map=struct_contact_map,
            real_struct_node_size=real_struct_node_size,
            embedding_info=embedding_info,
            embedding_attention_mask=embedding_attention_mask,
            label=None
        )
    )  # 将输入特征添加到特征列表
    batch_input = {}  # 初始化批量输入字典
    # "labels": torch.tensor([f.label for f in features], dtype=torch.long).to(args.device),
    # "labels": torch.tensor([f.label for f in features], dtype=torch.long).to(args.device),
    if seq_tokenizer:  # 如果有序列分词器
        batch_input.update(
            {
                "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long).to(args.device),
                "attention_mask": torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(args.device),
                "token_type_ids": torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(args.device),
            }
        )  # 更新批量输入字典
    if struct_tokenizer:  # 如果有结构分词器
        batch_input.update(
            {
                "struct_input_ids": torch.tensor([f.struct_input_ids for f in features], dtype=torch.long).to(args.device),
                "struct_contact_map": torch.tensor([f.struct_contact_map for f in features], dtype=torch.long).to(args.device),
            }
        )  # 更新批量输入字典
    if args.embedding_type:  # 如果有嵌入类型
        batch_input["embedding_info"] = torch.tensor(np.array([f.embedding_info for f in features], dtype=np.float32), dtype=torch.float32).to(args.device)  # 更新批量输入字典
        if args.embedding_type != "bos":  # 如果嵌入类型不为BOS
            batch_input["embedding_attention_mask"] = torch.tensor([f.embedding_attention_mask for f in features], dtype=torch.long).to(args.device)  # 更新批量输入字典

    return batch_info, batch_input  # 返回批量信息和批量输入

def predict_probs(
        args,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        model,
        row
):  # 定义predict_probs函数，用于预测概率
    '''
    prediction for one sample
    对一个样本进行预测
    :param args: 参数
    :param seq_tokenizer: 序列分词器
    :param subword: 子词
    :param struct_tokenizer: 结构分词器
    :param model: 模型
    :param row: 一个样本
    :return: 返回批量信息和概率
    '''
    '''
    label_list = processor.get_labels(label_filepath=args.label_filepath)
    label_map = {label: i for i, label in enumerate(label_list)}
    '''
    # in order to be able to embed longer sequences
    # 为了能够嵌入更长的序列
    model.to(torch.device("cpu"))  # 将模型移动到CPU
    batch_info, batch_input = transform_sample_2_feature(args, row, seq_tokenizer, subword, struct_tokenizer)  # 将样本转换为特征
    model.to(args.device)  # 将模型移动到设备
    if torch.cuda.is_available():  # 如果CUDA可用
        probs = model(**batch_input)[1].detach().cpu().numpy()  # 获取概率
    else:
        probs = model(**batch_input)[1].detach().numpy()  # 获取概率
    return batch_info, probs  # 返回批量信息和概率

def predict_binary_class(
        args,
        label_id_2_name,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        model,
        row
):  # 定义predict_binary_class函数，用于预测二分类
    '''
    predict positive or negative label for one sample
    预测一个样本的正负标签
    :param args: 参数
    :param label_id_2_name: 标签ID到名称的映射
    :param seq_tokenizer: 序列分词器
    :param subword: 子词
    :param struct_tokenizer: 结构分词器
    :param model: 模型
    :param row: 一个样本
    :return: 返回预测结果
    '''
    batch_info, probs = predict_probs(args, seq_tokenizer, subword, struct_tokenizer, model, row)  # 预测概率
    # print("probs dim: ", probs.ndim)
    preds = (probs >= args.threshold).astype(int).flatten()  # 获取预测结果
    res = []  # 初始化结果列表
    for idx, info in enumerate(batch_info):  # 遍历批量信息
        cur_res = [info[0], info[1], float(probs[idx][0]), label_id_2_name[preds[idx]]]  # 构造当前结果
        if len(info) > 2:  # 如果信息长度大于2
            cur_res += info[2:]  # 添加额外信息
        res.append(cur_res)  # 将当前结果添加到结果列表
    return res  # 返回结果

def predict_multi_class(
        args,
        label_id_2_name,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        model,
        row
):  # 定义predict_multi_class函数，用于预测多分类
    '''
    predict multi-labels for one sample
    预测一个样本的多标签
    :param args: 参数
    :param label_id_2_name: 标签ID到名称的映射
    :param seq_tokenizer: 序列分词器
    :param subword: 子词
    :param struct_tokenizer: 结构分词器
    :param model: 模型
    :param row: 一个样本
    :return: 返回预测结果
    '''
    batch_info, probs = predict_probs(args, seq_tokenizer, subword, struct_tokenizer, model, row)  # 预测概率
    # print("probs dim: ", probs.ndim)
    preds = np.argmax(probs, axis=-1)  # 获取预测结果
    res = []  # 初始化结果列表
    for idx, info in enumerate(batch_info):  # 遍历批量信息
        cur_res = [info[0], info[1], float(probs[idx][preds[idx]]), label_id_2_name[preds[idx]]]  # 构造当前结果
        if len(info) > 2:  # 如果信息长度大于2
            cur_res += info[2:]  # 添加额外信息
        res.append(cur_res)  # 将当前结果添加到结果列表
    return res  # 返回结果

def predict_multi_label(
        args,
        label_id_2_name,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        model,
        row
):  # 定义predict_multi_label函数，用于预测多标签
    '''
    predict multi-labels for one sample
    预测一个样本的多标签
    :param args: 参数
    :param label_id_2_name: 标签ID到名称的映射
    :param seq_tokenizer: 序列分词器
    :param subword: 子词
    :param struct_tokenizer: 结构分词器
    :param model: 模型
    :param row: 一个样本
    :return: 返回预测结果
    '''
    batch_info, probs = predict_probs(args, seq_tokenizer, subword, struct_tokenizer, model, row)  # 预测概率
    # print("probs dim: ", probs.ndim)
    preds = relevant_indexes((probs >= args.threshold).astype(int))  # 获取预测结果
    res = []  # 初始化结果列表
    for idx, info in enumerate(batch_info):  # 遍历批量信息
        cur_res = [
            info[0],
            info[1],
            [float(probs[idx][label_index]) for label_index in preds[idx]],
            [label_id_2_name[label_index] for label_index in preds[idx]]
        ]  # 构造当前结果
        if len(info) > 2:  # 如果信息长度大于2
            cur_res += info[2:]  # 添加额外信息
        res.append(cur_res)  # 将当前结果添加到结果列表
    return res  # 返回结果

def main():  # 定义main函数
    parser = argparse.ArgumentParser(description="Prediction RdRP")  # 创建参数解析器
    # for llm
    # 对于llm
    parser.add_argument(
        "--torch_hub_dir",
        default=None,
        type=str,
        help="set the torch hub dir path for saving pretrained model(default:~/.cache/torch/hub)"
    )  # 添加参数
    # for input
    # 对于输入
    parser.add_argument(
        "--fasta_file",
        default=None,
        type=str,
        required=True,
        help="fasta file path"
    )  # 添加参数
    parser.add_argument(
        "--save_file",
        default=None,
        type=str,
        required=True,
        help="the result file path"
    )  # 添加参数
    parser.add_argument(
        "--truncation_seq_length",
        default=4096,
        type=int,
        required=True,
        help="truncation seq length(include: [CLS] and [SEP]"
    )  # 添加参数
    parser.add_argument(
        "--emb_dir",
        default=None,
        type=str,
        help="the llm embedding save dir. default: None"
    )  # 添加参数
    parser.add_argument(
        "--pdb_dir",
        default=None,
        type=str,
        help="the 3d-structure pdb save dir. default: None"
    )  # 添加参数
    parser.add_argument(
        "--chain",
        default=None,
        type=str,
        help="pdb chain for contact map computing"
    )  # 添加参数

    # for trained checkpoint
    # 对于训练好的检查点
    parser.add_argument(
        "--dataset_name",
        default="rdrp_40_extend",
        type=str,
        required=True,
        help="the dataset name for model building."
    )  # 添加参数
    parser.add_argument(
        "--dataset_type",
        default="protein",
        type=str,
        required=True,
        help="the dataset type for model building."
    )  # 添加参数
    parser.add_argument(
        "--task_type",
        default=None,
        type=str,
        required=True,
        choices=["multi_label", "multi_class", "binary_class"],
        help="the task type for model building."
    )  # 添加参数
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="the model type."
    )  # 添加参数
    parser.add_argument(
        "--time_str",
        default=None,
        type=str,
        required=True,
        help="the running time string(yyyymmddHimiss) of trained checkpoint building."
    )  # 添加参数
    parser.add_argument(
        "--step",
        default=None,
        type=str,
        required=True,
        help="the training global step of model finalization."
    )  # 添加参数
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="sigmoid threshold for binary-class or multi-label classification, None for multi-class classification, defualt: 0.5."
    )  # 添加参数
    parser.add_argument(
        "--print_per_number",
        default=100,
        type=int,
        help="print per number"
    )  # 添加参数
    parser.add_argument(
        "--gpu_id",
        default=None,
        type=int,
        help="the used gpu index, -1 for cpu"
    )  # 添加参数
    input_args = parser.parse_args()  # 解析参数
    return input_args  # 返回参数

if __name__ == "__main__":
    args = main()  # 获取参数
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取脚本目录
    if args.torch_hub_dir is not None:  # 如果有torch_hub_dir
        if not os.path.exists(args.torch_hub_dir):  # 如果torch_hub_dir不存在
            os.makedirs(args.torch_hub_dir)  # 创建torch_hub_dir
        os.environ['TORCH_HOME'] = args.torch_hub_dir  # 设置TORCH_HOME环境变量
    if not os.path.exists(args.fasta_file):  # 如果fasta文件不存在
        print("the input fasta file: %s not exists!" % args.fasta_file)  # 打印错误信息
    if os.path.exists(args.save_file):  # 如果保存文件存在
        print("the output file: %s exists!" % args.save_file)  # 打印错误信息
    else:
        dirpath = os.path.dirname(args.save_file)  # 获取保存文件目录
        if not os.path.exists(dirpath):  # 如果保存文件目录不存在
            os.makedirs(dirpath)  # 创建保存文件目录
    model_dir = "%s/../models/%s/%s/%s/%s/%s/%s" % (
        SCRIPT_DIR, args.dataset_name, args.dataset_type, args.task_type,
        args.model_type, args.time_str,
        args.step if args.step == "best" else "checkpoint-{}".format(args.step)
    )  # 构造模型目录
    config_dir = "%s/../logs/%s/%s/%s/%s/%s" % (
        SCRIPT_DIR, args.dataset_name, args.dataset_type, args.task_type,
        args.model_type, args.time_str
    )  # 构造配置目录
    predict_dir = "%s/../predicts/%s/%s/%s/%s/%s/%s" % (
        SCRIPT_DIR, args.dataset_name, args.dataset_type, args.task_type,
        args.model_type, args.time_str,
        args.step if args.step == "best" else "checkpoint-{}".format(args.step)
    )  # 构造预测目录

    # Step1: loading the model configuration
    # 第一步：加载模型配置
    config = load_args(config_dir)  # 加载配置
    for key, value in config.items():  # 遍历配置项
        try:
            if value.startswith("../"):  # 如果值以"../"开头
                value = os.path.join(SCRIPT_DIR, value)  # 构造路径
        except AttributeError:
            continue
        print(f'My item {value} is labelled {key}')  # 打印配置项
        config[key] = value  # 更新配置项
    print("-" * 25 + "config:" + "-" * 25)  # 打印配置
    print(config)  # 打印配置
    print("-" * 60)  # 打印分隔线
    if config:  # 如果配置不为空
        args.dataset_name = config["dataset_name"]  # 设置数据集名称
        args.dataset_type = config["dataset_type"]  # 设置数据集类型
        args.task_type = config["task_type"]  # 设置任务类型
        args.model_type = config["model_type"]  # 设置模型类型
        args.has_seq_encoder = config["has_seq_encoder"]  # 设置是否有序列编码器
        args.has_struct_encoder = config["has_struct_encoder"]  # 设置是否有结构编码器
        args.has_embedding_encoder = config["has_embedding_encoder"]  # 设置是否有嵌入编码器
        args.subword = config["subword"]  # 设置子词
        args.codes_file = config["codes_file"]  # 设置子词代码文件
        args.input_mode = config["input_mode"]  # 设置输入模式
        args.label_filepath = config["label_filepath"]  # 设置标签文件路径
        if not os.path.exists(args.label_filepath):  # 如果标签文件路径不存在
            args.label_filepath = os.path.join(config_dir, "label.txt")  # 构造标签文件路径
        args.output_dir = config["output_dir"]  # 设置输出目录
        args.config_path = config["config_path"]  # 设置配置路径

        args.do_lower_case = config["do_lower_case"]  # 设置是否小写
        args.sigmoid = config["sigmoid"]  # 设置是否使用sigmoid
        args.loss_type = config["loss_type"]  # 设置损失类型
        args.output_mode = config["output_mode"]  # 设置输出模式

        args.seq_vocab_path = config["seq_vocab_path"]  # 设置序列词汇路径
        args.seq_pooling_type = config["seq_pooling_type"]  # 设置序列池化类型
        args.seq_max_length = config["seq_max_length"]  # 设置序列最大长度
        args.struct_vocab_path = config["struct_vocab_path"]  # 设置结构词汇路径
        args.struct_max_length = config["struct_max_length"]  # 设置结构最大长度
        args.struct_pooling_type = config["struct_pooling_type"]  # 设置结构池化类型
        args.trunc_type = config["trunc_type"]  # 设置截断类型
        args.no_position_embeddings = config["no_position_embeddings"]  # 设置是否无位置嵌入
        args.no_token_type_embeddings = config["no_token_type_embeddings"]  # 设置是否无标记类型嵌入
        args.cmap_type = config["cmap_type"]  # 设置接触图类型
        args.cmap_type = float(config["cmap_thresh"])  # 设置接触图阈值
        args.embedding_input_size = config["embedding_input_size"]  # 设置嵌入输入大小
        args.embedding_pooling_type = config["embedding_pooling_type"]  # 设置嵌入池化类型
        args.embedding_max_length = config["embedding_max_length"]  # 设置嵌入最大长度
        args.embedding_type = config["embedding_type"]  # 设置嵌入类型
        if args.task_type in ["multi-label", "multi_label"]:  # 如果任务类型为多标签
            args.sigmoid = True  # 设置sigmoid为True
        elif args.task_type in ["binary-class", "binary_class"]:  # 如果任务类型为二分类
            args.sigmoid = True  # 设置sigmoid为True

    if args.gpu_id <= -1:  # 如果GPU ID小于等于-1
        args.device = torch.device("cpu")  # 设置设备为CPU
    else:
        args.device = torch.device("cuda:%d" % args.gpu_id) if torch.cuda.is_available() else torch.device("cpu")  # 设置设备为GPU或CPU

    print("-" * 25 + "args:" + "-" * 25)  # 打印参数
    print(args.__dict__.items())  # 打印参数
    print("-" * 60)  # 打印分隔线
    '''
    print("-" * 25 + "model_dir list:" + "-" * 25)
    print(os.listdir(model_dir))
    print("-" * 60)
    '''

    if args.device.type == 'cpu':  # 如果设备类型为CPU
        print("Running Device is CPU!")  # 打印设备信息
    else:
        print("Running Device is GPU(%d)!" % args.gpu_id)  # 打印设备信息
    print("-" * 60)  # 打印分隔线

    # Step2: loading the tokenizer and model
    # 第二步：加载分词器和模型
    config, subword, seq_tokenizer, struct_tokenizer, model, label_id_2_name, label_name_2_id = \
        load_model(args=args, model_dir=model_dir)  # 加载模型
    predict_func = None  # 初始化预测函数
    if args.task_type in ["multi-label", "multi_label"]:  # 如果任务类型为多标签
        predict_func = predict_multi_label  # 设置预测函数为predict_multi_label
    elif args.task_type in ["binary-class", "binary_class"]:  # 如果任务类型为二分类
        predict_func = predict_binary_class  # 设置预测函数为predict_binary_class
    elif args.task_type in ["multi-class", "multi_class"]:  # 如果任务类型为多分类
        predict_func = predict_multi_class  # 设置预测函数为predict_multi_class
    else:
        raise Exception("Not Support Task Type: %s" % args.task_type)  # 抛出异常
    done = 0  # 初始化完成数量
    with open(args.save_file, "w") as wfp:  # 打开保存文件
        writer = csv.writer(wfp)  # 创建CSV写入器
        writer.writerow(["protein_id", "seq", "prob", "label"])  # 写入表头
        for row in fasta_reader(args.fasta_file):  # 遍历fasta文件
            # Step 3: prediction
            # 第三步：预测
            row = [row[0], clean_seq(row[0], row[1])]  # 清理序列
            res = predict_func(args, label_id_2_name, seq_tokenizer, subword, struct_tokenizer, model, row)  # 预测结果
            writer.writerow(res[0])  # 写入结果
            done += 1  # 增加完成数量
            if done % args.print_per_number == 0:  # 如果完成数量是打印间隔的倍数
                print("done : %d" % done)  # 打印完成数量
    print("all done: %d" % done)  # 打印总完成数量



