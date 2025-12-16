# Week3的工作内容：  
1. [阅读ProtTrans论文](https://ieeexplore.ieee.org/document/9477085/)
2. 阅读GitHub代码并复现 [ProtTrans](https://github.com/agemagician/ProtTrans)，分为如图几个流程去学习。
![pipeline](image.png)
3. 总结如何阅读代码，分为几个方面。
4. 总结如何阅读论文，从文章主旨入手，结合代码理解论文内容。并根据历史局限性来思考文章优缺点。

# 文件说明

本目录下文件及其作用：

- `0.1-prott5_embedder.py`：生成 ProtT5 embedding 的脚本（轻量化脚本，用于批量处理序列并输出 embedding）。
- `0.2prott5_embedder.ipynb`：交互式笔记本，包含示例运行与可视化，用于验证 embedding 流程。
- `0.3-prott5_embedding.py`：完整的 embedding 管道脚本（整合加载、预处理、推理与保存）。
- `test.fa`：用于快速验证的小规模 fasta 测试序列文件。

