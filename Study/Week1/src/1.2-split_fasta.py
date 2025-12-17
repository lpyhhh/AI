#!/usr/bin/env python3
# encoding: utf-8
'''
实现随机拆分 .fa 文件为训练集、验证集和测试集的功能。
#python ./src/split_fasta.py --input_csv /home/ec2-user/project/AI/Study/Week1/data/output.csv --output_train /home/ec2-user/project/AI/Study/Week1/data/train.csv --output_val /home/ec2-user/project/AI/Study/Week1/data/val.csv --output_test /home/ec2-user/project/AI/Study/Week1/data/test.csv
'''

import os
import csv
import random
import argparse

def split_fasta_file(input_csv, output_train, output_val, output_test, train_ratio, val_ratio, test_ratio):
    """
    将输入的 CSV 文件随机拆分为训练集、验证集和测试集，并分别保存到三个 CSV 文件中。

    参数：
        input_csv (str): 输入的 CSV 文件路径。
        output_train (str): 训练集输出 CSV 文件路径。
        output_val (str): 验证集输出 CSV 文件路径。
        output_test (str): 测试集输出 CSV 文件路径。
        train_ratio (float): 训练集的比例。
        val_ratio (float): 验证集的比例。
        test_ratio (float): 测试集的比例。
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为 1。"

    # 读取 CSV 文件中的所有行
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 读取表头
        rows = list(reader)  # 读取所有数据行

    # 随机打乱数据行顺序
    random.seed(42)
    random.shuffle(rows)

    # 计算拆分索引
    total_rows = len(rows)
    train_end = int(total_rows * train_ratio)
    val_end = train_end + int(total_rows * val_ratio)

    # 拆分数据行
    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]

    # 将拆分后的数据写入对应的 CSV 文件
    for output_csv, split_rows in zip(
        [output_train, output_val, output_test],
        [train_rows, val_rows, test_rows]
    ):
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # 写入表头
            writer.writerows(split_rows)  # 写入数据行

def main():
    """
    主函数：处理命令行参数并执行数据拆分。
    """
    parser = argparse.ArgumentParser(description="随机拆分 CSV 文件为训练集、验证集和测试集。")
    parser.add_argument("-i", "--input_csv", type=str, required=True, help="输入的 CSV 文件路径。")
    parser.add_argument("--output_train", type=str, required=True, help="训练集输出 CSV 文件路径（包含文件名）。")
    parser.add_argument("--output_val", type=str, required=True, help="验证集输出 CSV 文件路径（包含文件名）。")
    parser.add_argument("--output_test", type=str, required=True, help="测试集输出 CSV 文件路径（包含文件名）。")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集的比例 (默认: 0.7)。")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集的比例 (默认: 0.2)。")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集的比例 (默认: 0.1)。")

    args = parser.parse_args()

    # 验证比例是否正确
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("训练集、验证集和测试集的比例之和必须为 1。")

    # 拆分 CSV 文件
    split_fasta_file(
        args.input_csv,
        args.output_train,
        args.output_val,
        args.output_test,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )

if __name__ == "__main__":
    main()
