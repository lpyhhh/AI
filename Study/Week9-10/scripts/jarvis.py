import argparse
import torch
import os
from operator import itemgetter
from collections import deque
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import EsmTokenizer, EsmForSequenceClassification
from peft import PeftModel

# --- [模块 1] 生物模型接口 (The "Left Hand") ---
class BioPredictor:
    def __init__(self, lora_path):
        base_model_name = "facebook/esm2_t33_650M_UR50D"
        
        print(f"正在加载基础模型: {base_model_name}...")
        base_model = EsmForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=2,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        self.tokenizer = EsmTokenizer.from_pretrained(base_model_name)

        print(f"正在合并 LoRA 权重: {lora_path}...")
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # 匹配你在 3-lora.py 中的标签定义
        self.id2label = {0: "Negative (Non-CRESS)", 1: "Positive (CRESS Virus Rep Protein)"}
        print("生物模型加载完成。")

    # 【新增：修复 AttributeError 的核心函数】
    @torch.no_grad()
    def predict(self, sequence: str):
        """
        对蛋白质序列进行分类推理
        """
        # 1. 预处理：清洗序列并分词
        inputs = self.tokenizer(
            sequence.strip().upper(), 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        ).to(self.device)

        # 2. 模型推理
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # 3. 计算概率 (Softmax)
        # 使用 LaTeX 表达逻辑：$$P(i) = \frac{e^{z_i}}{\sum e^{z_j}}$$
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

        # 4. 返回识别出的标签名
        label = self.id2label.get(pred_idx, "Unknown")
        return f"{label} (置信度: {confidence:.2%})"

def build_agent_logic(db_path, llm_model_name):
    """
    将原本在 main() 里的 RAG 和 LLM 初始化逻辑提取出来
    """
    print(f"正在加载知识库: {db_path} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={'device': 'cuda'}
    )
    vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    print(f"正在加载 LLM: {llm_model_name} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, 
        max_new_tokens=1024, temperature=0.3, repetition_penalty=1.1, return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 修改 jarvis.py 中的 build_agent_logic 函数
    template = """<|im_start|>system
你是一个高级生物信息学助手，专门负责 CRESS 病毒和蛋白质序列分析。
你的任务是结合模型预测结果和提供的文献内容生成一份专业的报告。

# 任务指令：
1. 模型预测结果：{analysis_result}
2. 参考知识库内容：
{context}

# 要求：
- 如果知识库中有相关信息，请详细说明该预测结果的生物学背景。
- 如果知识库中没有直接相关的信息，请基于模型预测结果给出一般性建议。
- 必须回答用户的问题：{question}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        if not docs:
            print("⚠️ 警告：知识库检索结果为空！")
            return "未在知识库中找到相关参考资料。"
        
        print(f"📖 知识库成功检索到 {len(docs)} 条相关片段")
        return "\n\n".join([f"[来源: {d.metadata.get('source', '未知')}] {d.page_content}" for d in docs])

    # 构建 LCEL 链
    agent_chain = (
        {
            "context": itemgetter("search_query") | retriever | format_docs,
            "analysis_result": itemgetter("analysis_result"),
            "question": itemgetter("question")
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return agent_chain

# --- [模块 2] 修改后的 main 程序 ---
def main():
    parser = argparse.ArgumentParser(description="AI Bio-Agent: Sequence Analysis + Literature Search")
    parser.add_argument("--db_path", type=str, default="/home/ec2-user/project/Week5-6/paper_model")
    parser.add_argument("--bio_model_path", type=str, default="/home/ec2-user/project/Week3-4/results/model/final_lora_model")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    # 初始化左手
    bio_predictor = BioPredictor(args.bio_model_path)
    
    # 初始化大脑 (调用新提取的函数)
    agent_chain = build_agent_logic(args.db_path, args.llm_model)
    while True:
        user_input = input("\nUser (Input): ").strip()
        if user_input.lower() in ["exit", "quit"]: break
        
        # --- 智能路由逻辑 ---
        
        # 判断输入是否像生物序列 (简单的启发式规则)
        is_sequence = len(user_input) > 20 and not " " in user_input and \
                      (all(c in "ATCGU" for c in user_input.upper()) or \
                       all(c in "ACDEFGHIKLMNPQRSTVWY" for c in user_input.upper()))
        
        analysis_result = "N/A (用户未提供序列)"
        search_query = user_input # 默认搜用户的问题
        question = user_input
        
        # 在 jarvis.py 的 main() 或 API.py 的 ask_jarvis 中修改逻辑
        if is_sequence:
            print(">>> 🔬 检测到生物序列，启动分析引擎...")
            
            # 获取原始结果（包含置信度）
            full_pred = predictor.predict(user_input) 
            analysis_result = f"模型预测为: {full_pred}"
            
            # 【核心修改】：提取标签名称，去除置信度部分用于搜索
            # 假设 full_pred 是 "Positive (CRESS Virus Rep Protein) (置信度: 100.00%)"
            search_label = full_pred.split(" (置信度:")[0] 
            search_query = f"{search_label} characteristics and biological function"
            question = f"该序列已被预测为 {search_label}，请结合检索到的文献详细分析其生物学意义。"
        
        else:
            print(">>> 📖 检测到文本提问，启动检索模式...")

        # Step C: 调用大脑 (LLM + RAG)
        print(">>> 正在生成报告...")
        print("\nAssistant (A): ", end="", flush=True)
        
        try:
            # 流式输出
            for chunk in agent_chain.stream({
                "analysis_result": analysis_result,
                "search_query": search_query,
                "question": question
            }):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()