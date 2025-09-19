#/usr/bin/env python
'''
###3微调模型
#句子分类任务
#目的：check,model,seq,train_model
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer,AutoModelForSequenceClassification

checkpoint = "bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]

batch=tokenizer(sequences,padding=True,truncation=True,return_tensors="pt")
batch['labels']=torch.tensor([1,1])

optimizer=AdamW(model.parameters())
loss=model(**batch).loss
loss.backward()
optimizer.step()'''

###3.1 数据集加载，并清洗
#!pip install datasets
from datasets import load_dataset
from transformers import AutoTokenizer,DataCollatorWithPadding

checkpoint = "bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

raw_datasets=load_dataset("glue","mrpc")
raw_train=raw_datasets['train']

def tokenizer_funcation(example):
    return tokenizer(example['sentence1'],example['sentence2'],truncation=True)
tokenized_datasets=raw_datasets.map(tokenizer_funcation,batched=True)
#print(tokenized_datasets['train']['token_type_ids'])#句子分段

#padding
#实例化，操作
data_collator=DataCollatorWithPadding(tokenizer=tokenizer)#数据整理组合为batch
sample=tokenized_datasets['train'][:8]
sample={k:v for k,v in sample.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in sample['input_ids']]
#[50, 59, 47, 67, 59, 50, 62, 32]#token化后的文本形式


#pretrained
#超参数 模型 微调模型
from transformers import TrainingArguments,AutoModelForSequenceClassification,Trainer
training_args=TrainingArguments("test-trainer")
print(training_args)
model=AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=2)

trainer=Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()

#evaluate
#