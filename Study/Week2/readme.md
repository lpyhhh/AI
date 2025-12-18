# Week 3-4 学习计划：LoRA 微调 ESM-2
看什么：
oGitHub 搜索：peft esm2 fine-tuning（找代码）。
o重点看 LoraConfig 的设置。
做出来：
o一个 train_lora.py。
o关键代码填空：
Python
# L4 显卡支持 bf16 (BFloat16)，这比 fp16 更稳，一定要开！
training_args = TrainingArguments(
    ...,
    fp16=False,
    bf16=True,  # L4 的黑科技
    per_device_train_batch_size=8, # 24G 显存可以设大一点
    ...
)
达到目标：
o运行训练，用一个真实的数据集验证。lora微调后的时间和准确率都有提升。可以参考论文中如何设计验证模型优势的逻辑，去验证。
o基础理论上，可以说出来lora微调的原理，优势以及不足。如何进行微调的？



# 参考资料
英伟达框架：https://github.com/NVIDIA/bionemo-framework 
https://github.com/NVIDIA/bionemo-framework/tree/main/bionemo-recipes/models/esm2 主要介绍了基于 NVIDIA TransformerEngine 优化的 ESM-2 模型
LoraConfig 是 PeftConfig 的子类，用于指定 LoRA 微调的超参数。它定义了如何将 LoRA 适配器应用到模型的特定模块，以及 LoRA 的行为和特性

