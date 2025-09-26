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
from transformers import AutoTokenizer

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

#print(len(drug_dataset["train"].unique("drugName"))) # drugName

#test：使用 Dataset.unique() 函数查找训练和测试集中的特定药物和病症的数量 
#训练和测试集 药物数量，合并输出
drug_train=drug_dataset["train"]
drug_test=drug_dataset["test"]
sum_drug_train=set(drug_train.unique("drugName"))
sum_drug_test=set(drug_test.unique("drugName"))
#.unique返回列表 set()返回集合，无序不重复
sum_drug=len(sum_drug_test | sum_drug_train)
#print(sum_drug)

#2.4 删除空白行，并转化为小写
drug_dataset=drug_dataset.filter(lambda x:x["condition"] is not None)
def lowercase_condition(example):
    return {"condition":example["condition"].lower()}
drug_dataset.map(lowercase_condition)

#2.5 评论列拆分，统计长度输出新列，保留长度大于30
def compute_review_length(example):
    return{"review_length":len(example["review"].split())}
drug_dataset=drug_dataset.map(compute_review_length)

drug_dataset=drug_dataset.filter(lambda x:x["review_length"]>30)

#print(drug_dataset["train"].sort("review_length")[:1])

#2.6 html编码字符处理 review
import html
drug_dataset=drug_dataset.map(lambda x:{"review":html.unescape(x["review"])})

#2.7 Dataset.map() 方法有一个 batched 参数，加速处理数据
drug_dataset=drug_dataset.map(lambda x:{"review":[html.unescape(o) for o in x["review"]]},batched=True)

#2.8 token化，从一列特征提取多个特征
#return_overflowing_tokens=T 
#完全保留
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_and_split(example):
    result=tokenizer(
        example["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    sample_map = result.pop("overflow_to_sample_mapping") #这里面是超出部分的信息
    for key,value in example.items():
        result[key]=[value[i] for i in sample_map] #把键对应到超出部分的值上
    return result
tokenized_dataset =drug_dataset.map(tokenize_and_split,batched=True)

#2.9 数据格式转化
"""
Dataset.set_format() 展示方式
ds[:] 切片，数据底层改变
Dataset.from_pandas(df) 转为Dataset
"""
#计算 condition 列中不同类别的分布
train_df = drug_dataset["train"][:]
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "count": "frequency"})
)
#2.10 创建验证集
#拆分train给验证 train_test_split
drug_dataset_clean=drug_dataset["train"].train_test_split(train_size=0.8,seed=42)
drug_dataset_clean["validation"]=drug_dataset_clean.pop("test")
drug_dataset_clean["test"]=drug_dataset["test"]
#print(drug_dataset_clean)

#2.11 保存数据集 save_to_disk to_csv to_json ，读取load_from_disk
drug_dataset_clean.save_to_disk("目录")
