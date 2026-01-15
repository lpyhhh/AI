# 合并之前的 预测模型（week3-4）和大脑模型（week5-6 scripts/LLM.py），成为一个更完整的 RAG 模型（week9-10）

1. 核心逻辑架构
在 Agent 的世界里，LLM 是“大脑”，而你的 ESM-2 模型是“专科医生”。我们需要给这个医生套上一个标准的“听筒”（接口），让大脑知道什么时候该叫这位医生。

2. 环境准备
pip install torch transformers peft langchain langchain-core

3. 代码准备
为了保证效率，模型应该在 模块加载时或第一次调用时初始化一次（单例模式），而不是每次预测都重新加载权重。
python /home/ec2-user/project/Week9-10/scripts/jarvis.py \
    --db_path /home/ec2-user/project/Week5-6/paper_model \
    --bio_model_path /home/ec2-user/project/Week3-4/results/model/final_lora_model \
    --llm_model

知识点：
python跨代码调用方式：
- 直接 import 模块，必须要在同一个目录下，或者把模块路径加入 sys.path
- 使用 sys.path.append() 动态添加路径
from jarvis import analyze_cress_virus

