看什么：
o中文搜索：“HuggingFace Datasets 教程”。
o中文搜索：“Linux screen 命令用法”（防止断网导致训练中断）。
做出来：
o一个 data_prep.py 脚本。
o功能：读取你的 CSV，去掉长度超过 1024 的序列（L4 处理太长的序列会慢），把标签变成 0 和 1。

1.1 HuggingFace Datasets 数据加载与处理教程 数据加载，数据清洗，过滤，标签转换
比如说我现在有一些数据，把它处理成 lucaprot 可以使用的格式。
如何做？

1.2 lucaprot如何处理数据集  
![alt text](./image/image.png)

2.1 huggingface 大内存加载： 内存映射(memory-mapped) 文件来处理，解放内存管理问题；并通过 流式处理(streaming) 来摆脱硬盘限制。

2.2 lucaprot 加载

AI蛋白序列的长度有可能会超出模型训练的长度，用什么方法处理好呢？给我主流的方案以及出处
我要训练新模型，显存不够 参考 AlphaFold 3
引用点： "Spatial/Contiguous Cropping strategies during training."  做法： 在训练时随机截取序列片段（如 crop size = 384/512）进行训练，让模型学会局部特征；推理时利用显卡推理全长（或用滑动窗口推理）。这是研究生阶段最可行的方案。

我要处理多模态数据/结构数据	参考 ESM3	引用点： "Tokenization of continuous structures." 做法： 尝试将非序列数据（如结构坐标）离散化，减小输入维度。

我只是想跑通预测 (不想训练)	两者都不太好参考	AF3 和 ESM3 处理长序列主要靠暴力算力（80GB A100 GPU）

你的最佳方案依然是：滑动窗口 (Sliding Window)。

首选方案： Contiguous Cropping（连续随机裁剪）。将训练序列长度限制在 512 或 1024。这是性价比最高、最不容易报错的方案。

如果必须要长序列： 开启 Gradient Checkpointing。

如果显存还是不够： 配合使用 Gradient Accumulation（把 Batch Size 设为 1，累积 32 步更新一次）。

# 训练和推理，有什么不同？
训练是为了**“学”（很贵，费显存，策略是切碎学**）。推理是为了**“用”（便宜，省显存，策略是拼完整**）。

1.2 huggingface加载数据集

2 两种方法解决数据集太多问题，，然后如何导入到模型中运行？
