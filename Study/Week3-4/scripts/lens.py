import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import sys

# ===================== 1. 配置绘图样式 =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

# ===================== 2. 读取并清洗数据（核心修复：处理NaN/非数值） =====================
if len(sys.argv) < 2:
    print("用法：python lens.py <序列长度文件路径>")
    sys.exit(1)
data_path = sys.argv[1]

# 读取数据（先按字符串读取，避免类型错误）
df_raw = pd.read_csv(
    data_path,
    header=None,
    dtype=str  # 先按字符串读取，防止数值解析错误
)

# 数据清洗主逻辑
def clean_length_data(df_raw):
    # 情况1：两列（ID+长度）
    if df_raw.shape[1] >= 2:
        df = df_raw.iloc[:, [0, 1]].copy()  # 取前两列
        df.columns = ["seq_id", "length"]
        length_series = df["length"]
    # 情况2：一列（仅长度）
    else:
        df = df_raw.iloc[:, 0].copy()
        df.name = "length"
        length_series = df
    
    # 步骤1：清洗长度列 - 转为数值，无法转换的设为NaN
    length_series = pd.to_numeric(length_series, errors='coerce')
    
    # 步骤2：过滤NaN和非正数（长度不可能≤0）
    valid_lengths = length_series.dropna()
    valid_lengths = valid_lengths[valid_lengths > 0]
    
    # 输出清洗日志
    print("=== 数据清洗日志 ===")
    print(f"原始数据行数：{len(length_series)}")
    print(f"空值/非数值行数：{length_series.isna().sum()}")
    print(f"无效长度（≤0）行数：{len(length_series) - len(valid_lengths) - length_series.isna().sum()}")
    print(f"有效长度行数：{len(valid_lengths)}")
    
    return valid_lengths.values

# 执行清洗
lengths = clean_length_data(df_raw)

# 检查有效数据是否为空
if len(lengths) == 0:
    print("错误：无有效序列长度数据！请检查文件格式。")
    sys.exit(1)

# ===================== 3. 基础统计（仅基于有效数据） =====================
print("\n=== 序列长度统计信息 ===")
print(f"有效序列数：{len(lengths)}")
print(f"最短长度：{int(np.min(lengths))}")
print(f"最长长度：{int(np.max(lengths))}")
print(f"平均长度：{np.mean(lengths):.2f}")
print(f"中位数长度：{np.median(lengths):.2f}")

# ===================== 4. 优化：划分长度区间 =====================
# 自动计算合适的 Bins 数量，而不是固定步长
# 对于 150 万条数据，建议使用 50-100 个 Bin 左右
num_bins = 100
max_len = int(np.max(lengths))

# 排除极长序列（离群值）对绘图的影响（可选，比如只画到 99% 分位数）
upper_limit = np.percentile(lengths, 99) 
plot_lengths = lengths[lengths <= upper_limit]

# ===================== 5. 绘制直方图（数值型 X 轴） =====================
fig, ax = plt.subplots(figsize=(12, 7))

# 使用 ax.hist 直接处理数值，它会自动处理 X 轴刻度
n, bins, patches = ax.hist(
    plot_lengths, 
    bins=num_bins, 
    color="#2E86AB", 
    alpha=0.8, 
    edgecolor="white",
    linewidth=0.5
)

# 图表标注优化
ax.set_title(f"Sequence Length Distribution (Showing 99% data, Total: {len(lengths):,})", fontsize=14, pad=20)
ax.set_xlabel("Sequence Length (bp)", fontsize=12, labelpad=10)
ax.set_ylabel("Number of Sequences", fontsize=12, labelpad=10)

# 【关键修复】控制 X 轴刻度密度
# 让 Matplotlib 自动找寻最合适的 10-15 个刻度点
ax.xaxis.set_major_locator(MaxNLocator(nbins=15))

# 如果需要千分位分隔符
ax.get_yaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, p: format(int(x), ','))
)

# 添加数值标注（仅标注最高的几个柱子，避免杂乱）
max_count = n.max()
for i in range(len(n)):
    if n[i] > max_count * 0.05:  # 只标注高度超过最大值 5% 的柱子
        ax.text(
            bins[i] + (bins[i+1]-bins[i])/2, 
            n[i] + (max_count * 0.01), 
            f"{int(n[i]):,}", 
            ha='center', 
            fontsize=8, 
            rotation=0
        )

# 网格线
ax.grid(axis="y", linestyle="--", alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存+显示
plt.savefig("sequence_length_distribution.png", bbox_inches="tight")
plt.show()

# ===================== 6. 统计长度≤1024的序列 =====================
filtered_lengths = lengths[lengths <= 1024]
print(f"\n=== 长度≤1024的序列统计 ===")
print(f"数量：{len(filtered_lengths):,}")
print(f"占有效序列比例：{len(filtered_lengths)/len(lengths)*100:.2f}%")