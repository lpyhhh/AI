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
train_dataloader=DataLoader(tokenized_datasets['train'],shuffle=True,batch_size=8,collate_fn=data_collator)
eval_dataloader=DataLoader(tokenized_datasets['validation'],shuffle=True,batch_size=8,collate_fn=data_collator)
#检查数据性状
for batch in train_dataloader:
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

#学习率逐步降低，目的是为了防止 跳过最优点。
'''
区间：3 个 epochs × 每个 epoch 的 batch 数
学习率从 5e-5 线性降到 0，这个过程是在整个训练过程中完成的。
要算出“总共多少步”，就用 3 个 epochs × 每个 epoch 的 batch 数，这样调度器才知道“每一步”该把学习率降到多少。
'''
from torch.optim import AdamW
from transformers import get_scheduler

optimizer=AdamW(model.parameters(),lr=5e-5)#模型里所有可训练参数交给 AdamW

num_epochs=3
num_training_steps=num_epochs*len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",                       # 调度器类型：线性衰减
    optimizer=optimizer,            # 给哪个优化器调学习率
    num_warmup_steps=0,             # 热身步数，这里设 0 表示不需要
    num_training_steps=num_training_steps,  # 总步数（终点）
)
#步数print(num_training_steps)

### 训练
import torch
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
#print(device)
'''
导入tqdm
定义总轮数

训练模式
epoch循环
    batch循环
        batch数据
        model训练结果
        计算损失
        反向传播

        更新权重 + 调学习率 + 清零梯度
        进度条
'''
from tqdm import tqdm

progress_bar=tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch={k:v.to(device) for k,v in batch.items()}
        outputs=model(**batch)
        loss=outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

### 评估
# metric.compute() 方法
'''
eval评估

导入评测指标
模型eval评估模式
for eval数据集
    数据放到GPU
    关闭推理，只要output
    把logit转化为类别
    预测和真值保存
最终指标
'''
import evaluate

metric=evaluate.load("glue","mrpc")
model.eval()
for batch in eval_dataloader:
    batch={k:v.to(device) for k,v in batch.items()}
    with torch.no_grad():
        outputs=model(**batch)
    logits=output.logits
    predictions=torch.argmax(logits,dim=-1)
    metric.add_batch(predictions=predictions,references=batch["labels"])

print(metric.compute())

### 多设备运行
'''
accelerate加速计算类
导入函数，实例化
'''

#tesk SST-2数据集使用