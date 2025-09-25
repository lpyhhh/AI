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

data_files={
    "train":"/home/ec2-user/project/AI/data/SQuAD_it-train.json.gz",
    "test":"/home/ec2-user/project/AI/data/SQuAD_it-test.json.gz"
}
squad_it_dataset=load_dataset("json",data_files=data_files,field="data")
#field 如果不写 field="data"，库会把整份 JSON 当成一条样本返回
#split='train' 告诉库：“虽然只有一个文件，但请把它当成 train split 返回”。后续你可以用 dataset["train"] 来索引

###2 数据处理
from datasets import load_dataset

data_files = {
    "train": "/home/ec2-user/project/AI/data/drugsComTrain_raw.tsv", 
    "test": "drugsComTest_raw.tsv"}
drug_dataset=load_dataset("json",data_files=data_files,delimiter="\t")
print(drug_dataset)
#2.1 
