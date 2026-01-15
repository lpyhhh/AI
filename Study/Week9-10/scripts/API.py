import sys
import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 确保路径指向你存放 jarvis.py 的位置
sys.path.append("/home/ec2-user/project/Week9-10/scripts")
from jarvis import BioPredictor, build_agent_logic

# 1. 初始化 FastAPI
app = FastAPI(title="Jarvis Bio-API")

# 解决跨域问题，方便网页前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 全局模型单例加载 (在服务器启动时运行一次)
print("🚀 正在初始化全局模型，请稍候...")
BIO_MODEL_PATH = "/home/ec2-user/project/Week3-4/results/model/final_lora_model"
DB_PATH = "/home/ec2-user/project/Week5-6/paper_model"
LLM_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 这里的加载逻辑会触发你看到的那些日志
predictor = BioPredictor(BIO_MODEL_PATH)
agent_chain = build_agent_logic(DB_PATH, LLM_NAME)

# 定义输入数据模型
class BioQuery(BaseModel):
    text: str

# 3. 定义核心推理接口
@app.post("/ask")
async def ask_jarvis(query: BioQuery):
    user_input = query.text.strip()
    
    # 自动识别逻辑：判断是序列还是文本
    is_sequence = len(user_input) > 20 and not " " in user_input
    
    analysis_result = "N/A"
    search_query = user_input
    question = user_input
    
    try:
        if is_sequence:
            # 调用 ESM-2 LoRA 分类
            analysis_result = predictor.predict(user_input)
            search_query = f"{analysis_result} structure and function"
            question = f"该序列已被预测为 {analysis_result}，请结合文献分析其意义。"
        
        # 调用 RAG + LLM 链条
        # 注意：API 环境下通常使用 .invoke() 获取完整结果
        response = agent_chain.invoke({
            "analysis_result": analysis_result,
            "search_query": search_query,
            "question": question
        })
        
        return {
            "is_sequence": is_sequence,
            "prediction": analysis_result,
            "answer": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. 【关键步骤】启动阻塞式服务器
if __name__ == "__main__":
    # 使用 uvicorn 启动服务，监听 8000 端口
    # host="0.0.0.0" 允许外部 IP 访问你的 AWS 实例
    uvicorn.run(app, host="0.0.0.0", port=8000)