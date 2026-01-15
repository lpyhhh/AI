# 假设你的 pdb 叫 model.pdb，权重文件叫 structure_weights.txt
# 可以在 PyMOL 里的 File -> Run Script 运行下面这个逻辑
from pymol import cmd

# 读取权重
weights = []
with open("path/to/structure_weights.txt") as f:
    for line in f:
        weights.append(float(line.strip()))

# 逐个残基修改 B-factor
# 假设链名为 A
for i, w in enumerate(weights):
    # i+1 因为 PDB 残基编号通常从 1 开始
    cmd.alter(f"chain A and resi {i+1}", f"b={w}")

# 应用颜色映射
# 蓝色(低关注) -> 白色 -> 红色(高关注)
cmd.spectrum("b", "blue_white_red", "chain A")