import argparse
import sys
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def get_args():
    parser = argparse.ArgumentParser(
        description="Data Preprocessing for ESM-2: Length filtering and Amino Acid standardization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 输入输出文件
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file path")
    parser.add_argument("-o", "--output", required=True, help="Output cleaned FASTA file path")
    
    # 长度参数
    parser.add_argument("--min-len", type=int, default=60, help="Minimum sequence length to keep")
    parser.add_argument("--max-len", type=int, default=1024, help="Maximum sequence length to keep")
    
    # 清洗策略
    parser.add_argument("--strict", action="store_true", 
                        help="Strict mode: Convert B, Z, U, O to X. (If not set, keeps B, Z, U, O as they are known by ESM-2)")
    
    return parser.parse_args()

def clean_sequence(seq_str, strict_mode=True):
    """
    清洗序列逻辑：
    1. 转大写
    2. 移除末尾 '*'
    3. 根据模式替换非标准氨基酸
    """
    seq_str = seq_str.upper().rstrip("*")
    
    # 标准 20 种氨基酸
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
    # ESM-2 词表中包含 B, Z, U, O。
    # J 不在词表中，必须转 X。
    # 这里的 esm_allowed 取决于是否开启严格模式。
    if strict_mode:
        # 严格模式：除了标准20种，其他全部转X (包括 B, Z, U, O)
        allowed_set = standard_aa
    else:
        # 宽容模式：保留 ESM-2 认识的特殊氨基酸
        allowed_set = standard_aa.union(set("BZUO"))

    # 快速检查：如果全是合法字符，直接返回
    if set(seq_str) <= allowed_set:
        return seq_str, False # False 表示没有发生字符替换
    
    # 逐字符处理
    new_seq = []
    modified = False
    for char in seq_str:
        if char in allowed_set:
            new_seq.append(char)
        else:
            new_seq.append("X") # J 和其他乱码变成 X
            modified = True
            
    return "".join(new_seq), modified

def main():
    args = get_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"=== Processing Started ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Length Range: {args.min_len} - {args.max_len}")
    print(f"Strict Mode (B,Z,U,O -> X): {args.strict}")
    
    count_total = 0
    count_kept = 0
    count_too_short = 0
    count_too_long = 0
    count_cleaned_chars = 0 # 记录有多少条序列发生了字符替换(如 J->X)

    valid_records = []

    # 使用 Bio.SeqIO 读取 (内存高效，是 iterator)
    with open(args.output, "w") as output_handle:
        for record in SeqIO.parse(args.input, "fasta"):
            count_total += 1
            
            raw_seq = str(record.seq)
            
            # 1. 字符清洗
            cleaned_seq_str, is_modified = clean_sequence(raw_seq, strict_mode=args.strict)
            
            # 2. 长度过滤
            seq_len = len(cleaned_seq_str)
            
            if seq_len < args.min_len:
                count_too_short += 1
                continue
            if seq_len > args.max_len:
                count_too_long += 1
                continue
                
            # 3. 统计修改情况
            if is_modified:
                count_cleaned_chars += 1
            
            # 4. 写入文件
            # 重新构建 SeqRecord 对象
            new_record = SeqRecord(
                Seq(cleaned_seq_str),
                id=record.id,
                description=record.description
            )
            SeqIO.write(new_record, output_handle, "fasta")
            count_kept += 1
            
            # 简单的进度打印
            if count_total % 10000 == 0:
                print(f"Processed {count_total} sequences...", end="\r")

    print(f"\n\n=== Processing Complete ===")
    print(f"Total sequences processed: {count_total:,}")
    print(f"----------------------------------------")
    print(f"Discarded (Too short < {args.min_len}): {count_too_short:,}")
    print(f"Discarded (Too long > {args.max_len}): {count_too_long:,}")
    print(f"----------------------------------------")
    print(f"Sequences with special chars cleaned (-> X): {count_cleaned_chars:,}")
    print(f"----------------------------------------")
    print(f"Final valid sequences saved: {count_kept:,}")
    print(f"Output file: {args.output}")

if __name__ == "__main__":
    main()