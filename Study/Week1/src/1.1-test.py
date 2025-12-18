#!/usr/bin/env python3
# # encoding: utf-8
"""
input: fasta file
output: csv file (prot_id, seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename, label, and source.)
eg:like_YP_009351861.1_Menghai_flavivirus,MEQNG...,3416,,,,embedding_21449.pt,1,rdrp

pipeline:
process_fasta_file()  处理输入的 .fa 文件，并将数据保存到一个 CSV 文件中。

"""
import os
import csv
import argparse

def process_fasta_file(input_fa,output_csv):
    """
    input:
    output:
    """
    #定义csv 列头
    header = ["prot_id", "seq", "seq_len", "pdb_filename", "ptm", "mean_plddt", "emb_filename", "label", "source"]

    # opening fasta file while processing and save to csv file.
    with open(input_fa, "r") as fa_file, open(output_csv, mode="w", newline='',encoding='utf-8') as csvfile:
        # 流程：先处理id，去除> 保存到第一列；非>开头行，从上一个到下一个保存到seq列
        writer = csv.writer(csvfile) #开启写入模式
        writer.writerow(header)

        prot_id = None
        seq=""

        for line in fa_file:
            line = line.strip() # 换行  
            if line.startswith(">"):
                if prot_id and seq:
                    writer.writerow([prot_id, seq, len(seq), "", "", "", "", "", ""])
                prot_id = line[1:]
                seq = ""
            else:
                seq += line

        if prot_id and seq:
            writer.writerow([prot_id, seq, len(seq), "", "", "", "", "", ""])

def main():
    """

    """
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("-i", "--input", type=str, required=True, help="")
    parser.add_argument("-o", "--output", type=str, required=True, help="")

    args = parser.parse_args()

    process_fasta_file(args.input, args.output)

if __name__ == "__main__":
    main()

#process_fasta_file("/Users/pyl/Desktop/AI/rep_clean_id.fa","/Users/pyl/Desktop/AI/output.csv")
# python test.py -i /Users/pyl/Desktop/AI/rep_clean_id.fa -o /Users/pyl/Desktop/AI/output.csv