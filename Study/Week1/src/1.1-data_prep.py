#!/usr/bin/env python3
# encoding: utf-8
'''
输入: fasta文件
输出: csv文件 (prot_id, seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename, label, and source.)
eg:like_YP_009351861.1_Menghai_flavivirus,MEQNG...,3416,,,,embedding_21449.pt,1,rdrp
如果数据量过大，两个解决方案的原理：
    huggingface datasets库的内存映射和分块处理 
    lucaprot：if your dataset takes too much space to load into memory at once,use "src/data_process/data_preprocess_into_tfrecords_for_rdrp.py" to convert the dataset into "tfrecords". And create an index file: python -m tfrecord.tools.tfrecord2idx xxxx.tfrecords xxxx.index
'''

import os
import csv
import argparse

def process_fasta_file(input_file, output_csv):
    """
    处理输入的 .fa 文件，并将数据保存到一个 CSV 文件中。

    参数：
        input_file (str): 输入的 .fa 文件路径。
        output_csv (str): 输出 CSV 文件路径。
    """
    # 定义 CSV 文件的表头
    header = ["prot_id", "seq", "seq_len", "pdb_filename", "ptm", "mean_plddt", "emb_filename", "label", "source"]

    # 读取 .fa 文件中的所有序列
    with open(input_file, 'r') as fa_file, open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # 写入表头

        prot_id = None
        seq = ""

        for line in fa_file:
            line = line.strip()
            if line.startswith('>'):
                # 如果已经在处理一个序列，则将其保存到 CSV
                if prot_id and seq:
                    writer.writerow([prot_id, seq, len(seq), "", "", "", "", "", ""])

                # 开始处理一个新序列
                prot_id = line[1:]  # 去掉 '>' 字符
                seq = ""
            else:
                seq += line  # 追加序列行

        # 保存最后一个序列
        if prot_id and seq:
            writer.writerow([prot_id, seq, len(seq), "", "", "", "", "", ""])

def main():
    """
    主函数：处理命令行参数并执行数据处理。
    """
    parser = argparse.ArgumentParser(description="处理 .fa 文件并保存为 CSV 文件。")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="输入的 .fa 文件路径。")
    parser.add_argument("-o", "--output_csv", type=str, required=True, help="输出 CSV 文件路径（包含文件名）。")

    args = parser.parse_args()

    # 处理输入的 .fa 文件
    process_fasta_file(args.input_file, args.output_csv)

if __name__ == "__main__":
    main()
#python ./src/data_prep.py --input_file /home/ec2-user/project/AI/Study/Week1/data/rep_clean_id.fa --output_csv /home/ec2-user/project/AI/Study/Week1/data/output.csv