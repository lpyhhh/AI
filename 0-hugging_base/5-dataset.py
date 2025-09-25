#/usr/bin/env python

###1数据集加载
#1.1本地加载 远程加载
'''
导入包
导入数据（支持压缩包格式）
查看数据
'''
#wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz
#wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz
from datasets import load_dataset

data_files={"train": "/home/ec2-user/project/AI/data/SQuAD_it-train.json.gz", "test": "/home/ec2-user/project/AI/data/SQuAD_it-test.json.gz"}
datasets=load_dataset("json",data_files=data_files,field="data")
#print(datasets["train"][0]) 输出训练数据集中第一行数据内容

###2 处理数据的方法
#除了dataset.map()
#wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
#unzip drugsCom_raw.zip
data_files={"train":"/home/ec2-user/project/AI/data/drugsComTrain_raw.tsv",
            "test":"/home/ec2-user/project/AI/data/drugsComTest_raw.tsv"}
drug_dataset=load_dataset("csv",data_files=data_files,delimiter="\t")
#print(drug_dataset["train"][0])

#数据抽样检查 .shuffle() .select()
drug_sample=drug_dataset["train"].shuffle(seed=42).select(range(1000)) #42是随机种子， 1000是挑选的数量
#print(drug_sample[:3])
#data {'patient_id': 206461, 'drugName': 'Valsartan', 'condition': 'Left Ventricular Dysfunction', 'review': '"It has no side effect, I take it in combination of Bystolic 5 Mg and Fish Oil"', 'rating': 9.0, 'date': 'May 20, 2012', 'usefulCount': 27}

#2.1所有数据 id是否重复，与行数有关系？ Dataset.unique()
'''
drug_dataset.keys() dict_keys(['train', 'test'])
assert 如果条件为真，啥事没有；如果为假，立刻抛出 AssertionError 程序中断
'''
#for split in drug_dataset.keys():
#    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed"))

#2.2 列title重命名 rename_column
drug_dataset=drug_dataset.rename_columns({"Unnamed: 0":"patient_id"})
#这里列名改过之后，id重复查询 失效

#2.3 condition列 map() 处理，转化为小写
def lowercase_condition(example):
    return {"condition":example["condition"].lower()} #condition列的行返回
    
'''
drug_dataset.map(lowercase_condition)
#这样运行会报错：AttributeError:'NoneType' object has no attribute 'lower'
'''

#2.4 删除空的condition行
drug_dataset=drug_dataset.filter(lambda x:x['condition'] is not None)
#print(drug_dataset["train"][:1])

drug_dataset=drug_dataset.map(lowercase_condition)
#print(drug_dataset["train"]["condition"][:1])

#2.5 创建新列：对评论 操作
#空格 计算数量
def compute_review_length(example):
    return {"review_length":len(example["review"].split())} #返回新列 review_lenth
drug_dataset=drug_dataset.map(compute_review_length)

#2.6 排序
print(drug_dataset["train"].sort("review_length")[:3])

#2.7 删除review小于30的评论
drug_dataset=drug_dataset.filter(lambda x:x["review_length"]>30)
#print(drug_dataset.num_rows)

#2.8 解码对于html map

'''
当你的数据集不在 Hub 上时，你应该怎么做？
你如何切分和操作数据集？（如果你非常需要使用 Pandas，该如何处理？）
当你的数据集非常大，会撑爆你笔记本电脑的 RAM 时，你应该怎么办？
什么是“内存映射”和 “Apache Arrow”？
如何创建自己的数据集并将其推送到中心？
'''