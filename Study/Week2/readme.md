# Week 3-4 学习计划：LoRA 微调 ESM-2
看什么：
oGitHub 搜索：peft esm2 fine-tuning（找代码）。
o重点看 LoraConfig 的设置。
做出来：
o一个 train_lora.py。

# 达到目标：
理论基础 基础理论上，可以说出来lora微调的原理，优势以及不足。如何进行微调的？ 多看，多记，多回忆，多问。

实战经验 用一个真实的数据集验证。lora微调后的时间和准确率都有提升。可以参考论文中如何设计验证模型优势的逻辑，去验证。
LoraConfig lora peft微调esm2 查看未微调与调后的差异。

实战：
1 
# 参考资料
英伟达框架：https://github.com/NVIDIA/bionemo-framework
https://github.com/NVIDIA/bionemo-framework/tree/main/bionemo-recipes/models/esm2 主要介绍了基于 NVIDIA TransformerEngine 优化的 ESM-2 模型
LoraConfig 是 PeftConfig 的子类，用于指定 LoRA 微调的超参数。它定义了如何将 LoRA 适配器应用到模型的特定模块，以及 LoRA 的行为和特性