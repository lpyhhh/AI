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
'''
#超参数 模型 微调模型
from transformers import TrainingArguments,AutoModelForSequenceClassification,Trainer
training_args=TrainingArguments("test-trainer")
#print(training_args)
model=AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=2)

trainer=Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
#trainer.train()

#evaluate
import numpy as np
#接受对象，返回浮点数字典；predict()返回 “模型预测结果 + 真实标签 + 预测相关的统计指标”
predictions=trainer.predict(tokenized_datasets["validation"])
#[-0.39482936, -0.07467625]], dtype=float32), label_ids=array([1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1,
#预测结果选择最大值，label_ids是真实标签：如果用logits归一化后发现不对应，则结果不一定对
#print(predictions.predictions) #预测的结果

preds=np.argmax(predictions.predictions,axis=-1)
#print(preds)

#指标评估
# evaluate.load() 函数。返回的对象有一个 compute() 方法
'''

#3.2 数据处理
#评估函数， 模型训练
import numpy as np
import evaluate
from transformers import TrainingArguments,AutoModelForSequenceClassification,Trainer

def compute_metrics(eval_preds):
    metric=evaluate.load('glue','mrpc')
    logits,labels=eval_preds
    predications=np.argmax(logits,axis=-1)
    return metric.compute(predictions=predications,references=labels)

#model
training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
model=AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=2)

trainer=Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
#{'train_runtime': 117.6537, 'train_samples_per_second': 93.529, 'train_steps_per_second': 11.704, 'train_loss': 0.41183995265586704, 'epoch': 3.0}