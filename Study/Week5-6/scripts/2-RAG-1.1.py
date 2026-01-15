import argparse
import torch
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import deque

def main():
    parser = argparse.ArgumentParser(description="Chat with CRESS Knowledge Base")
    # 注意：这里不再需要 pdf_path，而是需要 db_path
    parser.add_argument("--db_path", type=str, default="vector_db/cress_index", help="向量库文件夹路径")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    # 1. 加载向量库 (秒级加载)
    print(f"正在加载向量库: {args.db_path} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={'device': 'cuda'}
    )
    # allow_dangerous_deserialization 是必须的，因为我们要加载本地文件
    vector_store = FAISS.load_local(args.db_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    # 2. 加载 LLM (复用之前的逻辑)
    print(f"正在加载 LLM: {args.model_id} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    text_gen_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, 
        max_new_tokens=1024, temperature=0.3, repetition_penalty=1.1, return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    # 3. 组装 Chain (同 v2.0，稍微优化提示词以适应多文档)
    template = """<|im_start|>system
你是一个CRESS病毒专家助手。所有回答必须基于【上下文】。
如果检索到的上下文包含多个来源，请综合它们的信息。
回答结束后，如果可能，请在括号里简要注明信息来源（例如：根据 paper1.pdf）。
<|im_end|>
<|im_start|>user
【历史】：{chat_history}
【上下文】：{context}
【问题】: {question}
<|im_end|>
<|im_start|>assistant
"""
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        # 把来源文件名也拼进去，方便模型知道是哪篇论文说的
        return "\n\n".join([f"[来源: {d.metadata.get('source', '未知')}] 内容: {d.page_content}" for d in docs])

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs, 
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt | llm | StrOutputParser()
    )

    # 4. 聊天循环
    print("\n✅ 百篇文献库已加载！开始提问。")
    memory_buffer = deque(maxlen=3)
    
    while True:
        query = input("\nUser (Q): ")
        if query.lower() in ["exit", "quit"]: break
        
        print("Assistant (A): ", end="", flush=True)
        history_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in memory_buffer])
        
        response = ""
        for chunk in rag_chain.stream({"question": query, "chat_history": history_str}):
            print(chunk, end="", flush=True)
            response += chunk
        print("\n")
        memory_buffer.append((query, response))

if __name__ == "__main__":
    main()