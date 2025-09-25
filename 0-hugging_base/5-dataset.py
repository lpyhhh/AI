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
    "test": "/home/ec2-user/project/AI/data/drugsComTest_raw.tsv"}
drug_dataset=load_dataset("csv",data_files=data_files,delimiter="\t")
#print(drug_dataset)

#2.1 查看小样本数据，随机抽样并看1000个 shuffle select
drug_sample=drug_dataset["train"].shuffle(seed=42).select(range(1000))
#print(drug_sample)

#2.2 查看Unnamed: 0'列 的行数和id是否相同 
#for split in drug_dataset.keys():
#    assert len(drug_dataset[split])==len(drug_dataset[split].unique("Unnamed: 0"))
#drug_dataset.keys() dict_keys(['train', 'test'])

#2.3 给Unnamed: 0列改名 patient_id  rename_columns()
drug_dataset=drug_dataset.rename_columns({"Unnamed: 0":"patient_id"})
#print(drug_dataset)

print(len(drug_dataset["train"].unique("drugName"))) # drugName

#test：使用 Dataset.unique() 函数查找训练和测试集中的特定药物和病症的数量 
#训练和测试集 药物数量，合并输出
drug_train=drug_dataset["train"]
drug_test=drug_dataset["test"]
sum_drug_train=set(drug_train.unique("drugName"))
sum_drug_test=set(drug_test.unique("drugName"))
#.unique返回列表 set()返回集合，无序不重复
sum_drug=len(sum_drug_test | sum_drug_train)
print(sum_drug)