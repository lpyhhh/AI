#!/usr/bin/env python3  # 指定解释器为 Python3
# encoding: utf-8  # 设置文件编码为 UTF-8
'''
输入: fasta文件  # 描述输入文件类型
输出: csv文件 (prot_id, seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename, label, and source.)  # 描述输出文件格式
eg:like_YP_009351861.1_Menghai_flavivirus,MEQNG...,3416,,,,embedding_21449.pt,1,rdrp  # 示例输出格式
如果数据量过大，两个解决方案的原理：  # 提供大数据量处理的解决方案
    huggingface datasets库的内存映射和分块处理  # 方案1：使用 huggingface datasets 库
    lucaprot：if your dataset takes too much space to load into memory at once,use "src/data_process/data_preprocess_into_tfrecords_for_rdrp.py" to convert the dataset into "tfrecords". And create an index file: python -m tfrecord.tools.tfrecord2idx xxxx.tfrecords xxxx.index  # 方案2：使用 lucaprot 转换为 tfrecords 格式
'''

import os  # 导入操作系统相关模块
import csv  # 导入 CSV 文件操作模块
import argparse  # 导入命令行参数解析模块

def process_fasta_file(input_file, output_csv):  # 定义函数处理 .fa 文件并保存为 CSV 文件
    """
    处理输入的 .fa 文件，并将数据保存到一个 CSV 文件中。  # 描述函数功能

    参数：  # 描述函数参数
        input_file (str): 输入的 .fa 文件路径。  # 输入文件路径
        output_csv (str): 输出 CSV 文件路径。  # 输出文件路径
    """
    # 定义 CSV 文件的表头
    header = ["prot_id", "seq", "seq_len", "pdb_filename", "ptm", "mean_plddt", "emb_filename", "label", "source"]  # 定义 CSV 文件的列名

    # 读取 .fa 文件中的所有序列
    with open(input_file, 'r') as fa_file, open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:  # 打开输入和输出文件 
        writer = csv.writer(csvfile)  # 创建 CSV 写入器
        writer.writerow(header)  # 写入表头

        prot_id = None  # 初始化蛋白质 ID
        seq = ""  # 初始化序列

        for line in fa_file:  # 遍历 .fa 文件的每一行
            line = line.strip()  # 去掉行首尾的空白字符
            if line.startswith('>'):  # 如果行以 '>' 开头，表示一个新序列的开始
                # 如果已经在处理一个序列，则将其保存到 CSV
                if prot_id and seq:  # 如果当前有蛋白质 ID 和序列
                    writer.writerow([prot_id, seq, len(seq), "", "", "", "", "", ""])  # 写入当前序列信息，第一次是空，不写入。第二次遍历时才写入。

                # 开始处理一个新序列
                prot_id = line[1:]  # 去掉 '>' 字符，获取蛋白质 ID，从第1个字符读取数据
                seq = ""  # 重置序列
            else:
                seq += line  # 追加序列行

        # 保存最后一个序列
        if prot_id and seq:  # 如果最后一个序列存在
            writer.writerow([prot_id, seq, len(seq), "", "", "", "", "", ""])  # 写入最后一个序列信息
        #为什么要多写一个if？因为写入内容，是读取下一个序列 才写入上一个

def main():  # 定义主函数
    """
    using argparse implement functionality
    主函数：处理命令行参数并执行数据处理。  # 描述主函数功能
    """
    parser = argparse.ArgumentParser(description="处理 .fa 文件并保存为 CSV 文件。")  # 创建命令行参数解析器
    parser.add_argument("-i", "--input_file", type=str, required=True, help="输入的 .fa 文件路径。")  # 添加输入文件参数
    parser.add_argument("-o", "--output_csv", type=str, required=True, help="输出 CSV 文件路径（包含文件名）。")  # 添加输出文件参数

    args = parser.parse_args()  # 解析命令行参数

    # 处理输入的 .fa 文件
    process_fasta_file(args.input_file, args.output_csv)  # 调用函数处理文件

if __name__ == "__main__":  # 如果当前模块是主模块
    main()  # 调用主函数
#python ./src/data_prep.py --input_file /home/ec2-user/project/AI/Study/Week1/data/rep_clean_id.fa --output_csv /home/ec2-user/project/AI/Study/Week1/data/output.csv  # 示例命令行调用

"""
open newline 正确的编码格式
"""