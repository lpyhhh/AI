# AI
这是一个AI学习路线图  
本仓库，默认你有**数学（微积分，概论，现代）基础知识**，考过研的那种。同时，看过**吴恩达老师的基础机器学习线性和非线性回归基础知识**。

## 学习路线1:  
0-hugging_base：[0-hugging_base](https://huggingface.co/learn/llm-course/zh-CN)
这是hugging face的基础模型介绍课程：  
该课程大概讲的内容为：直接使用包装好的库，进行微调和使用。同时也能了解如何使用hugging face。

## 学习路线2：生物模型构建以及测试

### 2： Transformer理解  
时间：**七天**
- 从零手写出 **最小可运行 Transformer（Encoder-Only）**
- 深入理解 **Self-Attention、Multi-Head Attention、Feed-Forward、Positional Encoding** 原理与实现细节
- 在 **IMDB 二分类任务**上完成训练并验证效果  

| 日 | 任务 | 预计时长 | 交付物 |
|---|---|---|---|
| Day1 | 精读 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) 博客，或者[知乎大神](https://zhuanlan.zhihu.com/p/75591049) + 做笔记 | 2h | 笔记.md |
| Day2 | 通读《动手学深度学习》Transformer 章节，推导 Self-Attention 公式 | 2h | 手写推导图 |
| Day3 | 用 PyTorch 实现 **Scaled Dot-Product Attention** + Mask 单元测试 | 3h | attention.py + 单元测试 |
| Day4 | 实现 **Multi-Head Attention** 与 **Positional Encoding**；组装 **Encoder Block** | 3h | encoder_block.py |
| Day5 | 实现 **Feed-Forward + LayerNorm + Residual**；堆叠 N 层 Encoder；构建分类头 | 3h | mini_transformer.py |
| Day6 | 加载 IMDB 数据集（HF datasets）、构建数据管道、编写训练循环 | 3h | train_imdb.py |
| Day7 | 训练 & 调参 & 可视化结果；撰写 README 与架构图 | 2h | mini-transformer.ipynb + README.md |

---
