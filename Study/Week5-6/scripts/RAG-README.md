# RAG (Retrieval-Augmented Generation) 学习笔记
https://gemini.google.com/app/4c734308be96f704 
本文档旨在记录学习 RAG 流程的完整步骤，从数据准备到最终生成答案。

## 什么是 RAG？

RAG 是一种结合了**检索 (Retrieval)** 和**生成 (Generation)** 模型的框架。它首先从一个大型知识库中检索与用户问题相关的文档片段，然后将这些片段作为上下文信息，连同用户的问题一起提供给一个大型语言模型（LLM），由 LLM 生成最终的、更准确、更有依据的答案。

## RAG 核心流程

RAG 的实现可以分为两个主要阶段：

1.  **数据索引阶段（离线）**: 准备知识库，将其处理成可供快速检索的格式。
2.  **检索生成阶段（在线）**: 当用户提问时，利用索引好的数据来生成答案。

---

### 阶段一：数据索引 (Indexing)

这是预处理步骤，目的是将你的文档资料库转换成一个向量数据库。

#### 1. 加载数据 (Loading)

首先，需要从各种来源加载你的文档。数据来源可以是：
- PDF 文件
- TXT 文件
- Markdown 文件
- 网页内容
- 数据库记录
- ...

**常用工具**: `LangChain` 提供了多种 `DocumentLoader` 来处理不同格式的文件。


#### 2. 文本分割 (Splitting)

由于 LLM 的上下文窗口长度有限，需要将长文档分割成更小的、语义完整的块 (Chunks)。

**常用策略**:
- 按字符数分割
- 递归按字符分割 (Recursive Character Splitting)
- 按 Markdown 标题或代码块分割

**常用工具**: `LangChain` 的 `TextSplitter`。


#### 3. 文本嵌入 (Embedding)

将每个文本块转换成一个数值向量（Embedding）。这个向量能够捕捉文本的语义信息。相似的文本会有相似的向量表示。

**常用模型**:
- `all-MiniLM-L6-v2` (Sentence-Transformers)
- `text-embedding-ada-002` (OpenAI)
- `bge-large-zh` (中文 embedding 模型)

**常用工具**: `Hugging Face Transformers`, `Sentence-Transformers`, `OpenAIEmbeddings`。

#### 4. 向量存储 (Storing)

将生成的文本块向量及其对应的原文内容存储到一个专门的向量数据库中。这个数据库支持高效的相似性搜索。

**常用向量数据库**:
- **本地**: `FAISS`, `ChromaDB`
- **云端**: `Pinecone`, `Weaviate`

---

### 阶段二：检索与生成 (Retrieval & Generation)

当用户提出问题时，执行以下步骤。

#### 5. 检索 (Retrieval)

接收用户的问题 (Query)，并执行以下操作：
1.  使用与索引阶段**相同的嵌入模型**将用户问题转换成一个向量。
2.  在向量数据库中搜索与问题向量最相似的 Top-K 个文本块向量。
3.  获取这些向量对应的原始文本块，它们将作为 LLM 的上下文。

#### 6. 生成 (Generation)

将检索到的上下文信息和用户的原始问题组合成一个提示 (Prompt)，然后将其发送给 LLM 以生成最终答案。

**Prompt 模板示例**:
```
请根据以下提供的上下文信息来回答用户的问题。
如果上下文中没有相关信息，就说你不知道。

上下文:
{context}

问题:
{question}

答案:
```

**常用工具**: `LangChain` 的 `RetrievalQA` 链可以自动化这个过程。

### 总结流程图

```
[原始文档] -> [加载器] -> [文档块] -> [文本分割器] -> [小块文本]
                                                          |
                                                          v
[嵌入模型] -> [向量] -> [向量数据库 (索引)]

-------------------- (以上为离线准备) --------------------
-------------------- (以下为在线查询) --------------------

[用户问题] -> [嵌入模型] -> [问题向量] -> [在向量数据库中搜索] -> [相关文本块]
                                                                  |
                                                                  v
[LLM] <- [Prompt (问题 + 相关文本块)] -> [最终答案]
```

# Myself
1 复现
conda install -c pytorch -c nvidia faiss-gpu
这个包需要从conda安装，因为版本问题，3.11不兼容faiss

回答乱，且内容多解决方案。（ “模型迷路” 现象）
模型太死板

为什么每次模型启动加载需要耗时比较长，如果部署上线，怎么解决？（向量化先，后加载）

1.2 构造cress向量数据库的文献，我怎么获得？爬取Google Scholar吗？(工程岗预备役，你要用**“API 思维”**而不是“爬虫思维”来解决问题)
三种方法，Google api，PubMed zotero，arxiv biorxiv
原则：文章质量有关；pdf格式要解析，

构造cress文献爬取，先向量化，后检索

2 代码
python /home/ec2-user/project/Week5-6/scripts/2.1-向量化.py \
    --data_dir /home/ec2-user/project/Week5-6/paper \
    --save_dir /home/ec2-user/project/Week5-6/paper_model \
python /home/ec2-user/project/Week5-6/scripts/2-RAG-1.1.py \
    --db_path /home/ec2-user/project/Week5-6/paper_model

3 知识点