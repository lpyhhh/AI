[cite\_start]这是一个为你定制的 **Week 2 专项练习题**。这份习题是根据你提供的学习路线中 Week 2 的核心内容（BERT/GPT 理解、Masked LM vs Autoregressive LM、HuggingFace 微调、Transformer 架构细节）设计的 [cite: 1]。

建议你先尝试在不看答案的情况下作答（尤其是代码部分，建议手写或在一个空白的 Notebook 中敲出来），以此来检验“手写所有相关代码”的目标是否达成。

-----

# Week 2 习题集：NLP 预训练模型与微调

## 第一部分：理论概念 (Conceptual Understanding)

**1. BERT 与 GPT 的核心区别**
根据你对 Encoder 和 Decoder 的理解，回答以下问题：

  * **A.** BERT 使用的是 Transformer 的哪一部分架构？（Encoder 还是 Decoder？）它是如何处理上下文信息的（单向还是双向）？请解释这种处理方式对文本分类任务的优势。
  * **B.** GPT 使用的是 Transformer 的哪一部分架构？它是如何处理上下文的？为什么说它更适合生成任务而不是单纯的理解任务？

**2. 预训练目标 (Pre-training Objectives)**

  * 简述 **MLM (Masked Language Model)** 的训练机制。例如：对于句子 `The cat sat on the [MASK].`，BERT 是怎么学习的？
  * 简述 **Autoregressive LM (自回归语言模型)** 的训练机制。这与 MLM 有何本质区别？

**3. Transformer 内部细节**
[cite\_start]根据 Week 2 笔记中提到的 Add\&Norm [cite: 1]：

  * **A.** 写出残差连接 (Residual Connection) 的数学公式。如果不使用残差连接，深层网络训练可能会遇到什么问题？
  * **B.** Transformer 中使用的是哪种归一化（Batch Norm, Layer Norm, 还是 Group Norm）？为什么在 NLP 任务中通常首选这种归一化方式，而不是图像处理中常用的 Batch Norm？

-----

## 第二部分：代码实战 (Coding Practice)

**4. HuggingFace Transformers 基础**
请补充以下代码片段，使用 HuggingFace `transformers` 库加载一个预训练的 BERT 模型用于文本分类任务（假设是二分类，如 IMDB）。

```python
from transformers import AutoTokenizer, ____________________

# 1. 加载预训练的 Tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 加载预训练的 BERT 模型用于序列分类
# 要求：指定分类数为 2 (IMDB positive/negative)
model = ____________________.from_pretrained(
    model_name, 
    ____________________ = 2  # 填空：设置标签数量的参数名
)

# 3. 简述：如果要查看模型的架构（例如查看每一层），你会使用什么 Python 命令？
# print(________)
```

**5. 数据处理与 Tokenization**
在处理 IMDB 数据输入 BERT 时，我们需要对文本进行 Tokenize。请写出调用 `tokenizer` 时通常需要的三个关键参数，并解释它们的作用。

  * 提示参数：`padding`, `truncation`, `max_length`。

<!-- end list -->

```python
inputs = tokenizer(
    "This movie was fantastic!",
    padding = __________,  # 解释作用
    truncation = __________, # 解释作用
    max_length = 512,
    return_tensors = "pt"
)
```

-----

## 第三部分：思考题 (Critical Thinking)

**6. 微调 (Fine-tuning) vs 预训练 (Pre-training)**

  * 在 Week 2 任务中，你需要在 IMDB 数据集上微调 BERT。请问：在微调阶段，我们通常是冻结（Freeze）BERT 的所有参数只训练最后一层分类层，还是对所有参数进行微小的更新（Full Fine-tuning）？这两种策略各有什么优缺点？

**7. Self-Attention 机制**

  * [cite\_start]笔记中提到“Token输入时，考虑两端所有数据的相关性”[cite: 1]。请用通俗的语言解释：在计算单词 "Bank" 的向量表示时，Self-Attention 机制是如何帮助模型区分 "River bank"（河岸）和 "Bank account"（银行账户）这两种不同含义的？

-----

## ✅ 参考答案与解析

### 第一部分：理论概念

**1. BERT vs GPT**

  * **A. BERT:** 使用 **Encoder**（编码器）。它是 **双向 (Bidirectional)** 的，即在编码一个 token 时，能同时看到它左边和右边的所有词。
      * *优势：* 对于文本分类（如 IMDB），模型需要理解整个句子的完整语义，双向视野能捕捉更丰富的上下文依赖。
  * **B. GPT:** 使用 **Decoder**（解码器）。它是 **单向 (Autoregressive/Left-to-Right)** 的，只能看到当前 token 之前的词。
      * *原因：* 生成任务是预测下一个词，不能“剧透”后面的词，所以必须单向。

**2. 预训练目标**

  * **MLM:** 随机 Mask 掉句子中的部分 Token（通常 15%），让模型根据上下文预测被遮盖的词。它强迫模型学习双向的上下文关系。
  * **Autoregressive:** 根据 $t$ 时刻之前的序列 $(x_1, ..., x_{t-1})$ 预测 $x_t$。本质是条件概率建模 $P(x_t | x_{<t})$。区别在于 MLM 看全局，AR 只看历史。

**3. Transformer 细节**

  * **A. 残差连接:** 公式为 $y = x + f(x)$（其中 $x$ 是输入，$f(x)$ 是该层的处理，如 Attention 或 FeedForward）。
      * *作用：* 解决**梯度消失**问题，允许梯度直接流向浅层，使深层网络更容易训练。
  * **B. 归一化:** 使用 **Layer Normalization (LN)**。
      * *原因：* Batch Norm 依赖于 batch size 且在变长序列（RNN/NLP）中效果不佳（不同句子长度统计量方差大）。Layer Norm 对每个样本内部的所有神经元进行归一化，不受 batch size 和序列长度影响。

### 第二部分：代码实战

**4. HuggingFace 填空**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ... (Tokenizer部分略)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels = 2 
)

# 查看架构
print(model)
```

**5. Tokenization 参数**

  * `padding=True` (或 `'max_length'`): 因为 Tensor 必须是矩阵形式，如果不等长，需要用 Pad token 补齐短句子，使其长度一致。
  * `truncation=True`: BERT 处理的最大长度有限（通常 512），超过长度的句子必须截断，否则报错。

### 第三部分：思考题

**6. 微调策略**

  * 通常做法是 **Full Fine-tuning**（更新所有参数），但使用很小的学习率（如 2e-5）。
  * *只训练分类层：* 训练快，显存占用少，但可能无法适配与预训练域差异大的新任务。
  * *全参数微调：* 效果通常最好，因为 BERT 内部的语义表示也会根据特定任务进行调整。

**7. Self-Attention 理解**

  * 当模型处理 "Bank" 时，Self-Attention 会计算 "Bank" 与句子中其他所有词（如 "River" 或 "Account"）的**相关性权重 (Attention Score)**。
  * 如果是 "River bank"，"River" 会获得较高的权重，模型将 "River" 的语义信息加权融合进 "Bank" 的向量中，从而将这个 "Bank" 编码为“河岸”的含义。

-----

**下一步建议：**
你现在的进度正处于 Week 2，如果你已经完成了上述题目并理解了背后的原理，我可以为你提供一个 **BERT 微调的最小化代码骨架 (Boilerplate code)**，帮助你快速完成 Week 2 的“手写所有相关代码”任务。需要吗？