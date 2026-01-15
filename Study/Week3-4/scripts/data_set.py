import argparse
import os
import random
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def get_args():
    parser = argparse.ArgumentParser(description="Simple Dataset Builder: Merge -> Random Split")
    
    # 输入文件
    parser.add_argument("--pos", required=True, help="Positive samples FASTA file")
    parser.add_argument("--neg", required=True, help="Negative samples FASTA file")
    
    # 输出设置
    parser.add_argument("--outdir", default="dataset_simple_split", help="Output directory")
    
    # 划分比例
    parser.add_argument("--split-ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="Train/Val/Test ratio (e.g. 0.8 0.1 0.1)")
    
    # 随机种子 (保证结果可复现)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    
    return parser.parse_args()

def tag_and_merge(pos_file, neg_file, output_fasta):
    """
    读取正负样本，添加前缀标签，并合并到一个列表中。
    """
    print(f"[1/3] Merging and tagging sequences...")
    records = []
    
    # 处理正样本
    print(f"  - Reading Positive: {pos_file}")
    pos_count = 0
    for record in SeqIO.parse(pos_file, "fasta"):
        # 修改 ID，添加 POS_ 前缀
        original_id = record.id
        record.id = f"POS_{original_id}"
        # 更新 description 防止 id 和 description 不一致
        record.description = record.description.replace(original_id, record.id)
        records.append(record)
        pos_count += 1
        
    # 处理负样本
    print(f"  - Reading Negative: {neg_file}")
    neg_count = 0
    for record in SeqIO.parse(neg_file, "fasta"):
        # 修改 ID，添加 NEG_ 前缀
        original_id = record.id
        record.id = f"NEG_{original_id}"
        record.description = record.description.replace(original_id, record.id)
        records.append(record)
        neg_count += 1
    
    # 保存一份全量备份
    SeqIO.write(records, output_fasta, "fasta")
    print(f"  -> Merged {len(records)} sequences saved to {output_fasta}")
    
    return records

def split_randomly(records, ratios, outdir):
    """
    简单随机划分
    """
    print(f"[2/3] Splitting dataset randomly (Ratio: {ratios})...")
    
    # 1. 随机打乱
    random.shuffle(records)
    
    total_seqs = len(records)
    r_train, r_val, r_test = ratios
    total_r = sum(ratios)
    
    # 2. 计算切分索引
    n_train = int(total_seqs * (r_train / total_r))
    n_val = int(total_seqs * (r_val / total_r))
    # 剩下的给 test
    
    train_set = records[:n_train]
    val_set = records[n_train : n_train + n_val]
    test_set = records[n_train + n_val:]
    
    # 3. 保存文件
    sets = {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }
    
    final_stats = {}
    
    for set_name, seq_list in sets.items():
        outfile = os.path.join(outdir, f"{set_name}.fasta")
        SeqIO.write(seq_list, outfile, "fasta")
        
        # 统计正负样本
        pos_cnt = sum(1 for r in seq_list if r.id.startswith("POS_"))
        neg_cnt = sum(1 for r in seq_list if r.id.startswith("NEG_"))
        final_stats[set_name] = (pos_cnt, neg_cnt)
        
        print(f"  -> {set_name.capitalize()}: {len(seq_list)} seqs saved to {outfile}")
        
    return final_stats

def main():
    args = get_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 准备目录
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        
    combined_fasta = os.path.join(args.outdir, "all_combined.fasta")
    
    # 1. 合并
    all_records = tag_and_merge(args.pos, args.neg, combined_fasta)
    
    # 2. 划分 (直接随机)
    stats = split_randomly(all_records, args.split_ratio, args.outdir)
    
    # 3. 报告
    print("\n" + "="*40)
    print("FINAL DATASET REPORT (Random Split)")
    print("="*40)
    print(f"{'Set':<10} {'Total':<10} {'Pos':<10} {'Neg':<10} {'Pos/Neg Ratio'}")
    print("-" * 55)
    
    for name in ["train", "val", "test"]:
        pos, neg = stats[name]
        total = pos + neg
        ratio = f"{pos/neg:.2f}" if neg > 0 else "inf"
        print(f"{name.upper():<10} {total:<10} {pos:<10} {neg:<10} {ratio}")
        
    print("="*40)
    print(f"Done. Outputs are in: {args.outdir}")

if __name__ == "__main__":
    main()