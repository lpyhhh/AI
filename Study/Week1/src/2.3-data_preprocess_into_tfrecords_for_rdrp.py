#!/usr/bin/env python
# encoding: utf-8
'''
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
@datetime: 2022/12/30 15:13  # 日期时间: 2022/12/30 15:13
@project: DeepProtFunc  # 项目: DeepProtFunc
@file: data_preprocess_into_tfrecords_for_rdrp  # 文件: data_preprocess_into_tfrecords_for_rdrp
@desc: transform the dataset for model building into tfrecords  # 描述: 将用于模型构建的数据集转换为tfrecords格式
'''
import os  # 导入os模块，用于操作系统相关功能
import random  # 导入random模块，用于生成随机数
import numpy as np  # 导入numpy模块，并简写为np，用于科学计算
import multiprocessing  # 导入multiprocessing模块，用于多进程处理
import tensorflow as tf  # 导入tensorflow模块，用于深度学习相关操作
import sys, csv, torch  # 导入sys、csv和torch模块
sys.path.append("..")  # 将上一级目录添加到系统路径
sys.path.append("../..")  # 将上两级目录添加到系统路径
sys.path.append("../../src")  # 将src目录添加到系统路径
try:
    from utils import write_fasta, fasta_reader, clean_seq, load_labels, file_reader, common_amino_acid_set  # 从utils模块导入多个函数和变量
except ImportError:
    from src.utils import write_fasta, fasta_reader, clean_seq, load_labels, file_reader, common_amino_acid_set  # 如果导入失败，从src.utils模块导入

class GenerateTFRecord(object):  # 定义GenerateTFRecord类，用于生成TFRecord文件
    def __init__(self, dataset_filename, structure_dir, embedding_dir, label_filepath, save_path, shuffle=False, num_shards=30):
        # 初始化类的属性
        self.shuffle = shuffle  # 是否打乱数据
        self.dataset_filename = dataset_filename  # 数据集文件名
        self.structure_dir = structure_dir  # 结构文件目录
        self.embedding_dir = embedding_dir  # 嵌入文件目录
        self.save_path = save_path  # 保存路径
        self.num_shards = num_shards  # 分片数量
        self.dataset = self.load_dataset(self.dataset_filename)  # 加载数据集
        self.prot_list = list(self.dataset.keys())  # 获取蛋白质ID列表
        if self.shuffle:  # 如果需要打乱数据
            for _ in range(5):
                random.shuffle(self.prot_list)  # 打乱蛋白质ID列表
        self.label_filepath = label_filepath  # 标签文件路径
        self.label_2_id = {label: idx for idx, label in enumerate(load_labels(self.label_filepath, header=True))}  # 加载标签并生成映射

        shard_size = (len(self.prot_list) + num_shards - 1)//num_shards  # 计算每个分片的大小
        indices = [(i * shard_size, (i + 1) * shard_size) for i in range(0, num_shards)]  # 生成分片索引
        indices[-1] = (indices[-1][0], len(self.prot_list))  # 调整最后一个分片的索引
        self.indices = indices  # 保存分片索引

    def load_dataset(self, header=True):  # 定义load_dataset函数，用于加载数据集
        '''
        load the dataset
        加载数据集
        :param header: whether contains header in the dataset file
        :return:
        '''
        dataset = {}  # 初始化数据集字典
        with open(self.dataset_filename, "r") as rfp:  # 打开数据集文件
            reader = csv.reader(rfp)  # 创建CSV读取器
            cnt = 0  # 初始化计数器
            for row in reader:  # 遍历每一行
                cnt += 1  # 增加计数器
                if cnt == 1 and header:  # 如果是第一行且包含表头
                    continue  # 跳过表头
                prot_id, seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename, label, source = row  # 解包行数据
                seq = seq.strip("*")  # 去掉序列中的*号
                dataset[prot_id] = [seq, None, None, label]  # 初始化数据集条目
                if self.structure_dir:  # 如果有结构文件目录
                    structure_filepath = os.path.join(self.structure_dir, pdb_filename)  # 构造结构文件路径
                    if os.path.exists(structure_filepath):  # 如果结构文件存在
                        dataset[prot_id][1] = structure_filepath  # 保存结构文件路径
                if self.embedding_dir:  # 如果有嵌入文件目录
                    embedding_filepath = os.path.join(self.embedding_dir, emb_filename)  # 构造嵌入文件路径
                    if os.path.exists(embedding_filepath):  # 如果嵌入文件存在
                        dataset[prot_id][2] = embedding_filepath  # 保存嵌入文件路径
                    else:
                        embedding_filepath = os.path.join(self.embedding_dir.replace("_append", ""), emb_filename)  # 替换路径中的_append后再尝试
                        if os.path.exists(embedding_filepath):  # 如果嵌入文件存在
                            dataset[prot_id][2] = embedding_filepath  # 保存嵌入文件路径
                        else:
                            print("%s emb filepath not exists!" % prot_id)  # 打印错误信息
        return dataset  # 返回数据集

    def _bytes_feature(self, value):  # 定义_bytes_feature函数，返回bytes_list类型的特征
        """Returns a bytes_list from a string / byte."""
        if not isinstance(value, list):  # 如果value不是列表
            value = [value]  # 转换为列表
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))  # 返回bytes_list特征

    def _float_feature(self, value):  # 定义_float_feature函数，返回float_list类型的特征
        """Returns a float_list from a float / double."""
        if not isinstance(value, list):  # 如果value不是列表
            value = [value]  # 转换为列表
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))  # 返回float_list特征

    def _dtype_feature(self):  # 定义_dtype_feature函数，返回int64_list类型的特征
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))  # 返回int64_list特征

    def _int_feature(self, value):  # 定义_int_feature函数，返回int64_list类型的特征
        if not isinstance(value, list):  # 如果value不是列表
            value = [value]  # 转换为列表
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))  # 返回int64_list特征

    def _serialize_example(self, obj_id, sequence, pdb_obj, embedding_obj, label):  # 定义_serialize_example函数，序列化样本
        d_feature = {'id': self._bytes_feature(obj_id.encode()), 'seq': self._bytes_feature(sequence.encode()),
                     'L': self._int_feature(len(sequence))}  # 初始化特征字典

        cur_example_label = label  # 获取当前样本的标签
        if cur_example_label is None or len(cur_example_label) == 0:  # 如果标签为空
            return None  # 返回None
        if isinstance(cur_example_label, list) or isinstance(cur_example_label, set):  # 如果标签是列表或集合
            cur_example_label_ids = [self.label_2_id[v] for v in cur_example_label]  # 获取标签ID
            d_feature['label'] = self._int_feature(cur_example_label_ids)  # 添加标签特征
        else:
            cur_example_label_id = self.label_2_id[cur_example_label]  # 获取标签ID
            d_feature['label'] = self._int_feature(cur_example_label_id)  # 添加标签特征

        if embedding_obj:  # 如果有嵌入对象
            d_feature['emb_l'] = self._int_feature(embedding_obj["L"][1])  # 添加嵌入长度特征
            d_feature['emb_size'] = self._int_feature(embedding_obj["d"][1])  # 添加嵌入大小特征
            for item in embedding_obj.items():  # 遍历嵌入对象的每一项
                name = item[0]  # 获取名称
                dtype = item[1][0]  # 获取数据类型
                value = item[1][1]  # 获取值
                if isinstance(value, np.ndarray):  # 如果值是numpy数组
                    value = list(value.reshape(-1))  # 转换为列表
                elif isinstance(value, int) or isinstance(value, float) or isinstance(value, str):  # 如果值是int、float或str
                    value = [value]  # 转换为列表
                if dtype == "str":  # 如果数据类型是字符串
                    d_feature[name] = self._bytes_feature(value)  # 添加bytes特征
                elif dtype == "int":  # 如果数据类型是整数
                    d_feature[name] = self._int_feature(value)  # 添加int特征
                else:
                    d_feature[name] = self._float_feature(value)  # 添加float特征
        if pdb_obj:  # 如果有结构对象
            d_feature['pdb_l'] = self._int_feature(pdb_obj["L"][1])  # 添加结构长度特征
            for item in pdb_obj.items():  # 遍历结构对象的每一项
                name = item[0]  # 获取名称
                dtype = item[1][0]  # 获取数据类型
                value = item[1][1]  # 获取值
                if isinstance(value, np.ndarray):  # 如果值是numpy数组
                    value = list(value.reshape(-1))  # 转换为列表
                elif isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
                    value = [value]
                if dtype == "str":
                    d_feature[name] = self._bytes_feature(value)
                elif dtype == "int":
                    d_feature[name] = self._int_feature(value)
                else:
                    d_feature[name] = self._float_feature(value)
        example = tf.train.Example(features=tf.train.Features(feature=d_feature))
        return example.SerializeToString()

    def _convert_numpy_folder(self, idx):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        tfrecord_fn = os.path.join(self.save_path, '%0.2d-of-%0.2d%s.tfrecords' % (idx + 1, self.num_shards, "_pdb_emb" if self.structure_dir and self.embedding_dir else ("_pdb" if self.structure_dir else "_emb" if self.embedding_dir else "")))
        # writer = tf.python_io.TFRecordWriter(tfrecord_fn)
        print("Save path:", tfrecord_fn)
        writer = tf.io.TFRecordWriter(tfrecord_fn)
        tmp_prot_list = self.prot_list[self.indices[idx][0]:self.indices[idx][1]]
        print("Serializing %d examples into %s" % (len(tmp_prot_list), tfrecord_fn))

        for i, protein_id in enumerate(tmp_prot_list):
            if i % 500 == 0:
                print("### Iter = %d/%d" % (i, len(tmp_prot_list)))
            item = self.dataset[protein_id]
            protein_seq = item[0]
            protein_label = item[-1]

            pdb_obj = None
            if item[1]:
                pdb_file = item[1]
                cmap = np.load(pdb_file, allow_pickle=True)
                ca_dist_matrix = cmap['C_alpha']
                cb_dist_matrix = cmap['C_beta']
                assert protein_seq == str(cmap['seqres'].item())
                assert protein_label == cmap['label'].item()
                pdb_obj = {
                    "L": ["int", ca_dist_matrix.shape[0]],
                    "C_alpha_dist_matrix": ["float", ca_dist_matrix],
                    "C_beta_dist_matrix": ["float", cb_dist_matrix]
                }
            embedding_obj = None
            if item[2]:
                embedding_file = item[2]
                embedding_obj = torch.load(embedding_file)
                # embeding_size
                bos_representations = embedding_obj["bos_representations"][36].numpy()
                # L * embeding_size
                representations = embedding_obj["representations"][36].numpy()
                # L * L
                contacts = embedding_obj["contacts"].numpy()
                if clean_seq(protein_id, protein_seq) != embedding_obj["seq"]:
                    print(set(protein_seq).difference(set(embedding_obj["seq"])))
                    print(set(embedding_obj["seq"]).difference(set(protein_seq)))

                embedding_obj = {
                    "L": ["int", representations.shape[0]], # representations.shape[0]
                    "d": ["int", representations.shape[1]],
                    "bos_representations": ["float", bos_representations],
                    "representations": ["float", representations],
                    # "contacts": ["float", contacts]
                }
            example = self._serialize_example(protein_id, protein_seq, pdb_obj, embedding_obj, protein_label)
            if example is None:
                continue
            writer.write(example)

        print("label size: %d" % len(self.label_2_id))
        print("Writing {} done!".format(tfrecord_fn))

    def run(self, num_threads):
        pool = multiprocessing.Pool(processes=num_threads)
        shards = [idx for idx in range(0, self.num_shards)]
        pool.map(self._convert_numpy_folder, shards)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    default="rdrp_extend_40",
    required=True,
    type=str,
    help="transform into tfrecords"
)
parser.add_argument(
    "--train",
    action="store_true",
    help="the dataset type"
)
args = parser.parse_args()

if __name__ == "__main__":
    if args.train:
        dataset_type_list = ["train"]
    else:
        dataset_type_list = ["train", "dev", "test"]
    num_shards = [1, 1, 1]
    for idx, dataset_type in enumerate(dataset_type_list):
        dataset_filename = "../dataset/%s/protein/binary_class/%s_with_pdb_emb.csv" % (args.dataset_name, dataset_type)
        structure_dir = None
        embedding_dir = "../dataset/%s/protein/binary_class/embs/" % args.dataset_name
        label_filepath = "../dataset/%s/protein/binary_class/label.txt" % args.dataset_name
        save_path = "../dataset/%s/protein/binary_class/tfrecords/%s/" % (args.dataset_name, dataset_type)
        tfr = GenerateTFRecord(dataset_filename,
                               structure_dir,
                               embedding_dir, label_filepath, save_path, shuffle=True if dataset_type == "train" else False, num_shards=num_shards[idx])
        tfr.run(num_threads=1)
    '''
    Note: need to build the index file, the cmd: python -m tfrecord.tools.tfrecord2idx 01-of-01_emb.tfrecords 01-of-01_emb.index
    '''
    try:
        from utils import file_reader, common_amino_acid_set
    except ImportError:
        from src.utils import file_reader, common_amino_acid_set
    not_common_seqs = []
    total = 0
    not_common = 0
    dataset_type_list = ["train", "dev", "test"]
    with open("../dataset/%s/protein/binary_class/contain_not_common_amino_acid.fasta" % args.dataset_name, "w") as wfp:
        for dataset_type in dataset_type_list:
            dataset_filename = "../dataset/%s/protein/binary_class/%s_with_pdb_emb.csv" % (args.dataset_name, dataset_type)
            for row in file_reader(dataset_filename, header=True, header_filter=True):
                prot_id, seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename, label, source = row
                diff = set(list(seq)).difference(common_amino_acid_set)
                total += 1
                if len(diff) > 0:
                    not_common += 1
                    wfp.write(prot_id + "\n")
                    wfp.write(seq + "\n")
                    wfp.write(str(diff) + "\n")
    print("%d, %d" % (total, not_common))








