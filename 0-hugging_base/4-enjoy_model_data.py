#/usr/bin/env python

#1
###使用模型 掩码填充（mask filling 
#pipeline是一个集合，直接做下游任务
#token和model 是细化后的任务
from transformers import pipeline
camembert_fill_mask =pipeline("fill-mask",model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
#print(results)

# 预训练模型时!!!
# 一定要检查它是如何训练的、在哪些数据集上训练的、它的局限性和偏见。所有这些信息都应在其模型卡片上有所展示。

#2
### 上传自己的模型到hugging
#训练时上传
