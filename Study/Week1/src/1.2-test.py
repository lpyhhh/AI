#!/usr/bin/env python3
# encoding: utf-8
"""
对 csv文件，拆分为 三个文件。

python /Users/pyl/Desktop/AI/Study/Week1/src/1.2-test.py --input /Users/pyl/Desktop/AI/output.csv --output_train /Users/pyl/Desktop/AI/train.csv --output_val /Users/pyl/Desktop/AI/val.csv --output_test /Users/pyl/Desktop/AI/test.csv

"""
import os
import argparse
import csv
import random

def split_fasta_file(input_csv,output_train,output_val,output_test,train_ratio,val_ratio,test_ratio):
    """
    判断和
    打开文件，读取文件，去除表头，读取所有行
    随机 顺序输入文件
    固定每个文件索引，
    open每个文件进行写入
    """
    assert abs(train_ratio+val_ratio+test_ratio - 1) < 1e-6
    
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile) # 错误：写成 input_csv
        header = next(reader)
        rows = list(reader)
        print(rows)

    random.seed(42)
    random.shuffle(rows)

    total_rows = len(rows)
    train_end = int(train_ratio * total_rows)
    val_end = train_end + int(total_rows * val_ratio)

    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]
    #print(test_rows)

    for output_csv,split_rows in zip(# 传参数，列表对应的输入
        [output_train, output_val, output_test],
        [train_rows, val_rows, test_rows]
    ):
        with open(output_csv,mode="w",newline="",encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(split_rows)

def main():
    """
    argparse input
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i","--input",type=str,required=True,help="")
    parser.add_argument("--output_train",type=str,required=True,help="")
    parser.add_argument("--output_val",type=str,required=True)
    parser.add_argument("--output_test",type=str,required=True)
    parser.add_argument("--train_ratio",type=float,default=0.7)
    parser.add_argument("--val_ratio",type=float,default=0.2)
    parser.add_argument("--test_ratio",type=float,default=0.1)

    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("分布概率不同")
    
    split_fasta_file(
        args.input,
        args.output_train,
        args.output_val,
        args.output_test,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )

if __name__ == "__main__":
    main()