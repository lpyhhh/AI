import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="将文件B的描述信息拼接到文件A的对应行末尾（基于第一列ID匹配）。"
    )
    parser.add_argument(
        '-a', '--input_a', 
        required=True, 
        help="输入文件 A (主文件)"
    )
    parser.add_argument(
        '-b', '--input_b', 
        required=True, 
        help="输入文件 B (提供注释信息)"
    )
    parser.add_argument(
        '-o', '--output', 
        required=True, 
        help="输出文件的路径"
    )
    return parser.parse_args()

def load_annotations(file_path):
    """
    读取文件 B，生成一个字典。
    Key: 第一列 ID
    Value: ID 后面的所有内容 (描述信息)
    """
    annotation_map = {}
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip():
                    continue
                
                # split(maxsplit=1) 只切分第一个空格
                # 例如: "ID123  description of protein" 
                # 变成: ["ID123", " description of protein"]
                parts = line.strip().split(maxsplit=1)
                
                if len(parts) == 2:
                    key_id = parts[0]
                    description = parts[1].strip()
                    annotation_map[key_id] = description
                elif len(parts) == 1:
                    # 只有ID没有描述的情况
                    annotation_map[parts[0]] = "No_Description"
                    
        print(f"[INFO] 已加载 {len(annotation_map)} 条注释信息来自文件 B。")
        return annotation_map
    except FileNotFoundError:
        print(f"[ERROR] 找不到文件 B: {file_path}")
        sys.exit(1)

def process_merge(file_a, annotation_map, output_file):
    """
    遍历文件 A，查找 ID 并拼接 B 的内容
    """
    match_count = 0
    total_count = 0
    
    try:
        with open(file_a, 'r', encoding='utf-8', errors='ignore') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                if not line.strip():
                    continue
                
                total_count += 1
                parts = line.strip().split()
                
                if parts:
                    a_id = parts[0] # 获取文件A的ID
                    
                    # 准备写入的内容
                    # strip() 去掉原行末尾换行符
                    base_line = line.strip() 
                    
                    if a_id in annotation_map:
                        # 拼接逻辑：原行 + Tab分隔符 + B的描述
                        # 使用 \t (Tab) 分隔，方便后续 Excel 或 pandas 读取
                        new_line = f"{base_line}\t{annotation_map[a_id]}\n"
                        f_out.write(new_line)
                        match_count += 1
                    else:
                        # 如果没匹配到，是丢弃还是保留？
                        # 通常注释操作会保留原行。如果只想保留匹配的，注释掉下面两行即可。
                        # f_out.write(base_line + "\n") # 若需保留未匹配行，请取消注释
                        pass # 这里为了符合你“匹配到”的语境，默认只输出匹配成功的行。

        print(f"[SUCCESS] 处理完成。")
        print(f" - 文件 A 总行数: {total_count}")
        print(f" - 成功匹配并合并: {match_count}")
        print(f"[INFO] 结果已保存至: {output_file}")

    except FileNotFoundError:
        print(f"[ERROR] 找不到文件 A: {file_a}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 发生未知错误: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    
    # 1. 加载注释库 (文件 B)
    anno_map = load_annotations(args.input_b)
    
    # 2. 处理文件 A 并合并输出
    process_merge(args.input_a, anno_map, args.output)

if __name__ == "__main__":
    main()