# ABO（AI+BIO）
这是一个AI+BIO基础流程学习路线，也就是用AI基础模型应用到生物行业。  
本仓库，默认你有**数学（微积分，概论，现代）基础知识**，考过研的那种。同时，看过**吴恩达老师的基础机器学习线性和非线性回归基础知识**。

## 学习路线1:  Base  
0-hugging_base：[0-hugging_base](https://huggingface.co/learn/llm-course/zh-CN)
这是hugging face的基础模型介绍课程：  
该课程大概讲的内容为：直接使用包装好的库，进行微调和使用。同时也能了解如何使用hugging face。

## 学习路线2：生物模型构建以及测试

### Week1： Transformer理解  
时间：**七天**
- 从零手写出 **最小可运行 Transformer**
- 深入理解 **Self-Attention、Multi-Head Attention、Feed-Forward、Positional Encoding** 原理与实现细节
- 在 **IMDB 二分类任务**上完成训练并验证效果，acc ≈ 0.9+

我的学习方法：  
1 搞通流程-ipynb  
2 自己复现-py

### Week2： 理解并使用BERT/GPT模型，微调  
时间：**七天**
- 理解：**masked language model（MLM）和 autoregressive LM**
- 微调：HuggingFace 进行 BERT 微调，完成**文本分类任务（如 IMDB）**
- 对比两个分类任务，encoder和decoder的不同
- 阅读两篇论文**BERT/GPT**  
相关资料：  
1 [BERT基础知识](https://datawhalechina.github.io/learn-nlp-with-transformers/#/)
2 [代码](https://chatgpt.com/c/68b6f63c-009c-832a-8d3f-26bd1346e7a6)[BERT/GPT 模型的具体实现方法]
论文