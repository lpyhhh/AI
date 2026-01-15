import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
# 
# ===================== 1. 配置绘图样式 =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei'] # 适配中英文
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 8) # 图表加宽一点
plt.rcParams['figure.dpi'] = 100

# ===================== 2. 读取并清洗数据 =====================
if len(sys.argv) < 2:
    print("用法：python script.py <序列长度文件路径>")
    # 如果没有命令行输入，用于测试的模拟数据（正式使用时请注释掉）
    # np.random.seed(42)
    # lengths = np.concatenate([
    #     np.random.normal(loc=300, scale=100, size=10000), # 主体
    #     np.random.exponential(scale=1000, size=1000)      # 长尾噪音
    # ])
    # lengths = lengths[lengths > 0]
    sys.exit(1)
else:
    data_path = sys.argv[1]
    print(f"正在读取文件: {data_path}...")
    # 读取数据（假设单列长度，如果是两列请参考之前的代码修改）
    try:
        df_raw = pd.read_csv(data_path, header=None, dtype=str)
        if df_raw.shape[1] >= 2:
             length_series = df_raw.iloc[:, 1]
        else:
             length_series = df_raw.iloc[:, 0]
             
        length_series = pd.to_numeric(length_series, errors='coerce')
        lengths = length_series.dropna()
        lengths = lengths[lengths > 0].values
        
        if len(lengths) == 0:
             print("错误：没有有效的序列长度数据。")
             sys.exit(1)
             
    except Exception as e:
        print(f"读取文件出错: {e}")
        sys.exit(1)

print(f"有效序列总数: {len(lengths):,}")

# ===================== 3. 计算正态分布统计量 (3σ原则) =====================
mu = np.mean(lengths)       # 均值
sigma = np.std(lengths)     # 标准差
k_sigma = 3                 # 设定为 3倍标准差

# 计算理论边界
lower_bound_calc = mu - k_sigma * sigma
upper_bound_calc = mu + k_sigma * sigma

# 实际应用边界（生物学限制：长度不能小于0，或者根据你的最小需求设定，如50）
actual_lower_bound = max(0, lower_bound_calc) 
actual_upper_bound = upper_bound_calc

# 统计保留与剔除的情况
kept_mask = (lengths >= actual_lower_bound) & (lengths <= actual_upper_bound)
kept_counts = np.sum(kept_mask)
excluded_counts = len(lengths) - kept_counts

print("\n=== 正态分布统计 (3σ) ===")
print(f"均值 (μ): {mu:.2f}")
print(f"标准差 (σ): {sigma:.2f}")
print(f"理论下界 (μ-3σ): {lower_bound_calc:.2f}")
print(f"理论上界 (μ+3σ): {upper_bound_calc:.2f}")
print("-" * 30)
print(f"实际应用保留区间: [{actual_lower_bound:.0f}, {actual_upper_bound:.0f}]")
print(f"保留数据量: {kept_counts:,} ({kept_counts/len(lengths)*100:.2f}%)")
print(f"剔除数据量: {excluded_counts:,}")

# ===================== 4. 可视化对比绘图 =====================
fig, ax = plt.subplots()

# --- A. 绘制原始分布直方图 ---
# 使用 'auto' 让 matplotlib 根据数据量自动决定 Bins 的数量和宽度
# color='gray' 设置为灰色背景，突出显示边界
n, bins, patches = ax.hist(
    lengths, 
    bins='auto',  
    color='#B0B0B0', 
    alpha=0.6, 
    edgecolor='none',
    label='Original Distribution (Raw Data)'
)

# --- B. 绘制统计边界线和区域 ---
# 1. 绘制均值线（蓝色实线）
ax.axvline(mu, color='blue', linestyle='-', linewidth=2, label=f'Mean (μ): {mu:.0f}')

# 2. 绘制下界和上界线（红色虚线）
# 如果下界是0，通常不需要画线，只画上界
if actual_lower_bound > 0:
    ax.axvline(actual_lower_bound, color='red', linestyle='--', linewidth=2)

ax.axvline(actual_upper_bound, color='red', linestyle='--', linewidth=2, label=f'3σ Bounds ({actual_lower_bound:.0f}-{actual_upper_bound:.0f})')

# 3. 【核心】用颜色区域填充来显示保留/剔除范围
# 填充保留区域（绿色）
ax.axvspan(actual_lower_bound, actual_upper_bound, color='green', alpha=0.15, label='Kept Region (within 3σ)')

# 填充剔除区域（红色，仅填充右侧长尾部分，左侧通常是0）
max_plot_x = bins[-1] # 获取当前X轴最大范围
if max_plot_x > actual_upper_bound:
    ax.axvspan(actual_upper_bound, max_plot_x, color='red', alpha=0.15, label='Excluded Region (>3σ)')

# --- C. 图表优化与标注 ---
# 【关键】设置Y轴为对数刻度。如果不设置，长尾部分根本看不见。
ax.set_yscale('log')
ax.set_ylabel('Count (Log Scale)', fontsize=12, labelpad=10)
ax.set_xlabel('Sequence Length (bp)', fontsize=12, labelpad=10)
ax.set_title(f'Sequence Length Distribution vs. 3σ Normal Bounds\n(Total: {len(lengths):,} seqs)', fontsize=14)

# 添加图例
ax.legend(loc='upper right', frameon=True, shadow=True)

# 在图上添加统计信息文本框
info_text = (
    f"Stats:\n"
    f"μ = {mu:.1f}\n"
    f"σ = {sigma:.1f}\n"
    f"3σ Range: [{actual_lower_bound:.0f}, {actual_upper_bound:.0f}]\n"
    f"Kept: {kept_counts/len(lengths)*100:.1f}%"
)
# 放置在图表右侧中部
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(1.02, 0.5, info_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', bbox=props)

ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.grid(axis='y', which='both', linestyle='--', alpha=0.3) # which='both' 对数轴也显示次级网格

plt.tight_layout()
# 调整布局以为右侧文本框留出空间
plt.subplots_adjust(right=0.85)

output_filename = "length_distribution_with_3sigma.png"
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"\n图表已保存为: {output_filename}")
plt.show()