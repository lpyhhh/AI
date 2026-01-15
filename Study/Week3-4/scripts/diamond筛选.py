import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="双向筛选：将蛋白分为两组。组A(保留)：所有比对E值都大于阈值；组B(剔除)：至少有一个比对E值小于阈值。"
    )
    parser.add_argument(
        '-i', '--input', 
        required=True, 
        help="输入文件 (Diamond 输出结果)"
    )
    parser.add_argument(
        '-o', '--output_kept', 
        required=True, 
        help="【保留文件】路径：存放那些 E-value 始终很差(大于阈值)的行。"
    )
    parser.add_argument(
        '-d', '--output_discarded', 
        required=True, 
        help="【剔除文件】路径：存放那些曾经出现过好匹配(小于阈值)的行。"
    )
    parser.add_argument(
        '-e', '--evalue', 
        type=float,
        required=True, 
        help="E-value 阈值 (例如 1e-5)。用于判断蛋白是否'已知'。"
    )
    return parser.parse_args()

def split_groups(input_file, file_kept, file_discarded, threshold):
    # 黑名单集合：存储那些“被判定为已知/好匹配”的第13列描述
    blacklist_descriptions = set()
    
    print(f"[1/2] 正在第一遍扫描：识别 E-value < {threshold} 的蛋白...")
    
    try:
        # --- 第一遍扫描：建立黑名单 ---
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 13:
                    continue

                try:
                    evalue = float(parts[10]) # 第11列
                    description = parts[12].strip() # 第13列
                    
                    # 只要发现有一行达标，加入黑名单
                    if evalue < threshold:
                        blacklist_descriptions.add(description)
                        
                except ValueError:
                    continue
        
        print(f"[INFO] 已识别 {len(blacklist_descriptions)} 个蛋白组属于'剔除/已知'类别。")

        # --- 第二遍扫描：分流输出 ---
        print(f"[2/2] 正在第二遍扫描：将数据分流到两个文件...")
        
        count_kept = 0
        count_discarded = 0
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(file_kept, 'w', encoding='utf-8') as f_keep, \
             open(file_discarded, 'w', encoding='utf-8') as f_disc:
            
            for line in f_in:
                if not line.strip():
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 13:
                    continue
                
                description = parts[12].strip()
                
                # --- 分流逻辑 ---
                if description in blacklist_descriptions:
                    # 如果在黑名单里，写入剔除文件
                    f_disc.write(line)
                    count_discarded += 1
                else:
                    # 如果不在黑名单里，写入保留文件
                    f_keep.write(line)
                    count_kept += 1

        print("-" * 30)
        print(f"[处理完成]")
        print(f"1. 保留文件 (差匹配/未知): {file_kept}")
        print(f"   -> 行数: {count_kept}")
        print(f"2. 剔除文件 (好匹配/已知): {file_discarded}")
        print(f"   -> 行数: {count_discarded}")
        print("-" * 30)

    except FileNotFoundError:
        print(f"[ERROR] 找不到输入文件: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 发生未知错误: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    split_groups(args.input, args.output_kept, args.output_discarded, args.evalue)

if __name__ == "__main__":
    main()