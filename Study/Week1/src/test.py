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

def process_fasta_file(input_fa,output_csv):
    """
    input:
    output:
    """
    #定义csv 列头
    header = ["prot_id", "seq", "seq_len", "pdb_filename", "ptm", "mean_plddt", "emb_filename", "label", "source"]

    # opening fasta file while processing and save to csv file.
    with open(input_fa, "r") as fa_file, open(output_csv, mode="w", new)