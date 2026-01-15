import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TextIteratorStreamer
from threading import Thread

def main():
    """
    代码目的:
    加载一个4位量化的大型语言模型(LLM)并进行对话
    """
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
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. 加载模型与 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/home/ec2-user/project/Week5-6/model"
    )
    # 打印模型加载完成信息
    print("\nModel loaded! Let's chat (Type 'exit' to quit).\n" + "-"*30)

    # 4. 对话循环
    # 初始化对话历史，包含一个系统角色的初始消息
    messages = [
        {"role": "system", "content": "你是一个乐于助人的生物学AI助手。"}
    ]

    while True:
        """
        1 初始化数据格式
        2 获取 用户输入并token化
        3 将用户输入内容格式化输入给模型分析，并输出
        """
        user_input = input("\nLPY: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break
        messages.append({"role": "user", "content": user_input})
        text = tokenizer.apply_chat_template(
            messages,
            tokenizer=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=args.max_tokens)

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