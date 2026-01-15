import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="关键词筛选工具：检查文件B的每一行是否包含文件A中的任意关键词。"
    )
    parser.add_argument(
        '-a', '--keywords', 
        required=True, 
        help="文件 A：包含关键词的列表 (例如: Cressdnaviricota)"
    )
    parser.add_argument(
        '-b', '--target_file', 
        required=True, 
        help="文件 B：被搜索的目标文件"
    )
    parser.add_argument(
        '-o', '--output', 
        required=True, 
        help="输出文件路径"
    )
    parser.add_argument(
        '--mode', 
        choices=['extract', 'merge'], 
        default='extract',
        help="模式选择 (非反选时有效): 'extract' (只输出B内容) 或 'merge' (输出 关键词+B内容)。"
    )
    parser.add_argument(
        '-v', '--invert', 
        action='store_true', 
        help="反选模式：输出【不包含】任何关键词的行。"
    )
    # --- 新增参数 ---
    parser.add_argument(
        '-i', '--ignore_case', 
        action='store_true', 
        help="忽略大小写：启用后 'Virus' 可以匹配 'virus'。"
    )
    return parser.parse_args()

def load_keywords(file_path):
    """加载关键词列表"""
    keywords = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                content = line.strip()
                if content:
                    keywords.append(content)
        print(f"[INFO] 已加载 {len(keywords)} 个关键词来自文件 A。")
        return keywords
    except FileNotFoundError:
        print(f"[ERROR] 找不到文件 A: {file_path}")
        sys.exit(1)

def search_and_write(file_b, keywords, output_file, mode, invert, ignore_case):
    """在文件B中搜索关键词，支持反选和忽略大小写"""
    match_count = 0
    
    try:
        with open(file_b, 'r', encoding='utf-8', errors='ignore') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                if not line.strip():
                    continue
                
                # 预处理行内容（如果忽略大小写，则转为小写用于比较）
                # 注意：我们只用 line_check 来比较，输出时还是用原版 line
                line_check = line.lower() if ignore_case else line
                
                found_keyword = None
                
                for key in keywords:
                    # 预处理关键词
                    key_check = key.lower() if ignore_case else key
                    
                    if key_check in line_check:
                        found_keyword = key # 记录原始关键词(用于merge模式输出)
                        break 
                
                # --- 逻辑判断部分 ---
                
                # 情况1: 反选模式 (-v) -> 没找到关键词才写入
                if invert:
                    if found_keyword is None:
                        f_out.write(line)
                        match_count += 1
                        
                # 情况2: 正常模式 -> 找到了关键词才写入
                else:
                    if found_keyword is not None:
                        if mode == 'extract':
                            f_out.write(line)
                        elif mode == 'merge':
                            # 即使忽略大小写匹配，输出时也用原始关键词 found_keyword
                            f_out.write(f"{found_keyword}\t{line}")
                        match_count += 1
                        
        print(f"[SUCCESS] 处理完成。")
        print(f" - 模式: {'反选' if invert else mode}")
        print(f" - 忽略大小写: {'是' if ignore_case else '否'}")
        print(f" - 结果行数: {match_count}")
        print(f"[INFO] 结果已保存至: {output_file}")

    except FileNotFoundError:
        print(f"[ERROR] 找不到文件 B: {file_b}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 发生未知错误: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    keys = load_keywords(args.keywords)
    search_and_write(
        args.target_file, 
        keys, 
        args.output, 
        args.mode, 
        args.invert,
        args.ignore_case # 传入新参数
    )

if __name__ == "__main__":
    main()