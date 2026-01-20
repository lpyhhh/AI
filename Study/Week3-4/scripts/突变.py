import argparse
import random
import sys
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# 标准20种氨基酸
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="通过随机点突变增强蛋白质序列数据 (Data Augmentation by Mutation)"
    )

    # 必需参数
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="输入 FASTA 文件路径 (例如: positives.fasta)"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="输出 FASTA 文件路径 (例如: positives_aug.fasta)"
    )

    # 可选参数
    parser.add_argument(
        "-n", "--num_copies", 
        type=int, 
        default=10, 
        help="增强倍数：每个原始序列生成多少个突变副本 (默认: 10)"
    )
    parser.add_argument(
        "-r", "--rate", 
        type=float, 
        default=0.03, 
        help="突变率：0.0到1.0之间 (默认: 0.03, 即3%%的位点发生突变)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="随机种子，保证结果可复现 (默认: 42)"
    )

    return parser.parse_args()

def mutate_sequence(sequence, rate):
    """
    对序列进行随机点突变
    """
    seq_list = list(sequence)
    length = len(seq_list)
    # 计算需要突变的位点数量
    num_mutations = int(length * rate)
    
    # 如果计算结果为0但rate>0，至少突变1个（可选策略，这里严格按比例）
    if num_mutations == 0 and rate > 0 and length > 0:
        num_mutations = 1

    # 随机选择要突变的位置
    mutation_indices = random.sample(range(length), num_mutations)
    
    for idx in mutation_indices:
        original_aa = seq_list[idx]
        # 确保突变成不同的氨基酸
        candidates = [aa for aa in AMINO_ACIDS if aa != original_aa]
        if candidates:
            new_aa = random.choice(candidates)
            seq_list[idx] = new_aa
        
    return "".join(seq_list)

def main():
    # 1. 解析参数
    args = parse_arguments()

    # 2. 设置随机种子 (为了实验可复现性)
    random.seed(args.seed)

    # 3. 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在。")
        sys.exit(1)

    augmented_records = []
    original_count = 0
    generated_count = 0

    print(f"[*] 开始处理: {args.input}")
    print(f"[*] 配置: 倍数={args.num_copies}, 突变率={args.rate}, 种子={args.seed}")

    try:
        for record in SeqIO.parse(args.input, "fasta"):
            original_count += 1
            seq_str = str(record.seq)

            # A. 首先保留原始序列
            augmented_records.append(record)

            # B. 生成突变副本
            for i in range(args.num_copies):
                mutated_seq = mutate_sequence(seq_str, args.rate)
                generated_count += 1
                
                new_id = f"{record.id}_aug_{i+1}"
                new_record = SeqRecord(
                    Seq(mutated_seq),
                    id=new_id,
                    description=f"mutated from {record.id} rate={args.rate}"
                )
                augmented_records.append(new_record)
        
        # 4. 保存结果
        if augmented_records:
            SeqIO.write(augmented_records, args.output, "fasta")
            print(f"\n[+] 处理完成！")
            print(f"    - 原始序列数: {original_count}")
            print(f"    - 生成突变数: {generated_count}")
            print(f"    - 总序列数: {len(augmented_records)}")
            print(f"    - 结果保存至: {args.output}")
        else:
            print("\n[!] 警告: 没有读取到任何序列，请检查输入文件格式。")

    except Exception as e:
        print(f"\n[!] 发生错误: {e}")

if __name__ == "__main__":
    main()