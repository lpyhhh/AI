本周计划:
安装一个 回答型的大模型 Llama

前期调研：打造一个知识回答型的gpt。现在序列识别已经完成，问答qwen-2.5B与知识库。

实战：
    1 千问3
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config, # 应用上面定义的4位量化配置
        device_map="auto",              # 自动将模型层分配到可用的硬件（GPU、CPU）上
        trust_remote_code=True,         # 允许执行模型仓库中的自定义代码
        cache_dir="/home/ec2-user/project/Week5-6/model" # 指定模型缓存目录
    )
复盘

问AI问题：
虚拟环境安装包要分开
我在ec2上部署，如何把他映射到端口，远程访问？


huggingface的readme看不懂
对于部署，我们推荐使用SGLang和vLLM等框架
对于本地使用，我们强烈推荐使用Ollama、LMStudio、MLX、llama.cpp和KTransformers等工具
本地使用：
    部署到本地的方式。
量化推理加速框架：
    旨在优化大型语言模型（LLM）的推理效率。例如说API访问，不用vLLM可能会卡死。
    比 vLLM 更新的框架，专门针对“复杂的连续提问”做了加速

# 1执行
## 2代码阅读
在构建模型时，BitsAndBytesConfig是啥？
    [text](https://zhuanlan.zhihu.com/p/665601576)