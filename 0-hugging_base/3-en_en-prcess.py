#/usr/bin/env python
'''
第三章前，全部的流程
'''
### 导入包 数据处理
from datasets import load_dataset
from transformers import AutoTokenizer,DataCollatorWithPadding

raw_datasets=load_dataset("glue","mrpc")
checkpoint="bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example['sentence1'],example['sentence2'],truncation=True)

tokenized_datasets=raw_datasets.map(tokenize_function, batched=True)#函数的另一种使用方式：tokenize_function()，Python 会先执行这个函数（但此时没有传入example 参数，会直接报错），这显然不符合需求。map先批量处理数据，再传递给函数
data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
#print(tokenized_datasets)#查看数据处理结果

#训练前数据清洗：指定我们想要的数据 删除 重命名 格式 查看 ["attention_mask", "input_ids", "labels", "token_type_ids"]
tokenized_datasets=tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets=tokenized_datasets.rename_column("label","labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names #查看处理后训练集包含的列名（相当于 “数据集的表头”）
#print(tokenized_datasets)

### 数据导入集设置
from torch.utils.data import DataLoader
test_dataloader=DataLoader(tokenized_datasets['train'],shuffle=True,batch_size=8,collate_fn=data_collator)
eval_dataloader=DataLoader(tokenized_datasets['validation'],shuffle=True,batch_size=8,collate_fn=data_collator)
#检查数据性状
for batch in test_dataloader:
    break
for k,v in batch.items():
    #print(v.shape)
    break
"""
{'attention_mask': torch.Size([8, 65]),
 'input_ids': torch.Size([8, 65]),
 'labels': torch.Size([8]),
 'token_type_ids': torch.Size([8, 65])}
为什么是([8, 65])？
8 每个batch大小， batch_size=8
65 padding长度
"""
### 模型构建
from transformers import AutoModelForSequenceClassification
model=AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=2)

output=model(**batch)
#print(output.loss,output.logits.shape)

### 训练模型
#优化率 学习率调度器（如何定义？如何使用）
#优化率 目的：根据损失函数让loss最小化。 用法：接受loss.backward()
#学习率调度器=步长
