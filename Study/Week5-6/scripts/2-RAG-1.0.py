import argparse
import torch
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline,HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from threading import Thread
from transformers import TextIteratorStreamer
from langchain_community.vectorstores import FAISS
from operator import itemgetter  # <--- 新增这行
from collections import deque # 新增：用于管理记忆队列
"""
python3 Week5-6/scripts/2-RAG-1.0.py --pdf_path /home/ec2-user/project/Week5-6/paper/cress-base.pdf
"""

def main():
    # --- 1. 参数设置 ---
    parser = argparse.ArgumentParser(description="RAG System: Chat with your PDF")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="LLM Model ID")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding Model ID")
    args = parser.parse_args()

    # --- 2. 处理 PDF 文档 ---
    print(f"[1/4] 正在读取文档: {args.pdf_path} ...")
    if not os.path.exists(args.pdf_path):
        print("错误：文件不存在！")
        return

    loader = PyPDFLoader(args.pdf_path)
    docs = loader.load()
    
    # [优化点1]：切分稍微变大一点，有助于获取完整概念
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    print(f"      文档已切分为 {len(splits)} 个片段。")

    # --- 3. 向量化与存储 ---
    print(f"[2/4] 正在建立向量索引 ({args.embed_model}) ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embed_model, 
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    vector_store = FAISS.from_documents(splits, embeddings)
    print("      向量库构建完成。")

    # --- 4. 加载 LLM ---
    print(f"[3/4] 正在加载 LLM: {args.model_id} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/home/ec2-user/project/Week5-6/model"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024, # 允许回答得更长
        return_full_text=False, 
        temperature=0.3, # 稍微提高温度，增加一点灵活性
        repetition_penalty=1.1,
        do_sample=True 
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # --- 5. 构建带记忆的 Prompt ---
    
    # [优化点2]：更新 Prompt 模板，加入 {chat_history} 占位符
    # 并且修改指令，允许适度解释
    template = """<|im_start|>system
你是一个专业的生物学研究助手。请结合【上下文】和【对话历史】回答用户的【问题】。
规则：
1. 优先从【上下文】中提取答案。
2. 如果【上下文】信息不足，但问题是关于生物学通用概念（如"什么是共进化"），请用你的专业知识补充解释。
3. 如果完全无法回答，再说"文档中未提及"。
<|im_end|>
<|im_start|>user
【对话历史】：
{chat_history}

【上下文】：
{context}

【问题】: {question}
<|im_end|>
<|im_start|>assistant
"""
    
    prompt = PromptTemplate.from_template(template)

    # [优化点3]：扩大搜索范围 k=6
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {
                # 1. 拿 input_data["question"] 去检索
                "context": itemgetter("question") | retriever | format_docs, 
                # 2. 拿 input_data["question"] 填入 Prompt 的 {question}
                "question": itemgetter("question"),
                # 3. 拿 input_data["chat_history"] 填入 Prompt 的 {chat_history}
                "chat_history": itemgetter("chat_history")
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    print("\n" + "="*50)
    print(" RAG 系统 (v2.0 增强版) 就绪！")
    print("="*50 + "\n")

    # [优化点4]：手动管理历史记录 (Sliding Window)
    # 只保留最近 3 轮对话，防止 Prompt 太长爆显存
    memory_buffer = deque(maxlen=3) 

    while True:
        query = input("\nUser (Q): ")
        if query.lower() in ["exit", "quit"]:
            break
            
        # 把队列里的历史记录拼成字符串
        history_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in memory_buffer])
        
        print("Assistant (A): ", end="", flush=True)
        
        full_response = ""
        try:
            # 必须显式传入 chat_history
            input_data = {"question": query, "chat_history": history_str}
            
            for chunk in rag_chain.stream(input_data):
                print(chunk, end="", flush=True)
                full_response += chunk
        except Exception as e:
            print(f"Error: {e}")
            
        print("\n")
        
        # 存入记忆
        memory_buffer.append((query, full_response))

if __name__ == "__main__":
    main()