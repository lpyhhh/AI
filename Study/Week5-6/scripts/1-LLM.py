#!/usr/bin/env python3
"""
代码目的:
加载一个4位量化的大型语言模型(LLM)并进行对话
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from threading import Thread
from transformers import TextIteratorStreamer

def main():
    # 1. 定义参数
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser(description="Load a 4-bit quantized LLM for chat.")
    # 添加模型ID参数，默认是 Qwen/Qwen2.5-7B-Instruct
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                        help="Hugging Face model ID (e.g., meta-llama/Meta-Llama-3-8B-Instruct or Qwen/Qwen2.5-7B-Instruct)")
    # 添加最大生成token数参数，默认是 512
    parser.add_argument("--max_tokens", type=int, default=512, help="Max new tokens to generate")
    # 解析命令行参数
    args = parser.parse_args()

    # 打印正在加载的模型信息
    print(f"Loading model: {args.model_id} in 4-bit...")

    # 2. 配置量化参数 (NF4)
    # 创建 BitsAndBytesConfig 用于4位量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # 启用4位加载
        bnb_4bit_quant_type="nf4",              # 设置量化类型为 nf4 (Normal Float 4)
        bnb_4bit_compute_dtype=torch.float16,   # 在计算过程中，权重会反量化为 float16 类型以进行矩阵乘法
        bnb_4bit_use_double_quant=True,         # 启用双重量化，进一步节省内存
    )

    # 3. 加载模型与 Tokenizer
    # 从预训练模型加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # 从预训练模型加载模型，并应用量化配置
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config, # 应用上面定义的4位量化配置
        device_map="auto",              # 自动将模型层分配到可用的硬件（GPU、CPU）上
        trust_remote_code=True,         # 允许执行模型仓库中的自定义代码
        cache_dir="/home/ec2-user/project/Week5-6/model" # 指定模型缓存目录
    )

    # 打印模型加载完成信息
    print("\nModel loaded! Let's chat (Type 'exit' to quit).\n" + "-"*30)

    # 4. 对话循环
    # 初始化对话历史，包含一个系统角色的初始消息
    messages = [
        {"role": "system", "content": "你是一个乐于助人的生物学AI助手。"}
    ]

    # 开始一个无限循环，用于持续对话
    while True:
        # 获取用户输入
        user_input = input("\nUser: ")
        # 如果用户输入 'exit' 或 'quit'，则退出循环
        if user_input.lower() in ["exit", "quit"]:
            break

        # 将用户的输入添加到对话历史中
        messages.append({"role": "user", "content": user_input})

        # 应用对话模板 (Chat Template)
        # 使用 tokenizer 将对话历史格式化为模型期望的输入字符串
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # 返回字符串而不是 token IDs
            add_generation_prompt=True  # 为助手角色添加生成提示
        )
        # 将格式化后的文本转换为 PyTorch 张量，并移动到模型所在的设备
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 流式输出 (Streamer) - 像 ChatGPT 一样打字
        # 创建一个 TextIteratorStreamer 用于流式生成文本
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        # 准备生成函数的参数，包括输入和 streamer
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=args.max_tokens)
        
        # 创建一个新线程来运行模型的 generate 方法，以避免阻塞主线程
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        print("Assistant: ", end="")
        response_text = ""
        # 遍历 streamer 以获取并打印生成的文本
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response_text += new_text
        print()

        # 将模型的完整回复添加到对话历史中
        messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()