#!/usr/bin/env bash
set -e

<<EOF
数据集：
标准cress rep蛋白
${DATA_DIR}/rep/rep.fa
长度处理后的ictv数据
${DATA_DIR}/proteins_m_M.fasta
ncbi下载所有rep序列
${DATA_DIR}/proteins_base_substituted.fasta ${DATA_DIR}/proteins_base_substituted.id
${DATA_DIR}/true/cress_lens.fasta 限定只有cress，有极高的假阳性，可以用来做最后新病毒发现的测试集

验证集：ICTV数据库中的rep蛋白序列
${RESULTS_DIR}/data/mmseq_rep_seq.fasta 17,410：1307 /home/ec2-user/project/results/test/true_verify/data/true_verify.fasta

训练集
正样本：${RESULTS_DIR}/data/positive_true_deduped.fasta 25,771:65,544
${RESULTS_DIR}/data/positive/pep.out.fa.90 2,252
负样本数据来源：环境样本中预测的蛋白序列，排除与ICTV rep蛋白有显著相似性的序列
${RESULTS_DIR}/data/negative/pep.out.fa.40 65,544

数据清洗流程
数据长度60-1024
碱基替换
正负样本挑选，正样本中包含所有的cress病毒序列，负样本与正样本相似性低。

优化数据集构造流程：
正样本：标准cress rep蛋白， 去冗余90%
划分数据集
EOF
########################## 1.0 version
## 数据目录
DATA_DIR="/home/lpy/hdd/myself/AI/data"
SCRIPTS_DIR="/home/lpy/hdd/myself/AI/scripts"
RESULTS_DIR="/home/lpy/hdd/myself/AI/results"
THREADS=192
# 1 数据集划分
## 1.1 正样本 ${DATA_DIR}/true/rep.90.fa
### 正：银标准 seqkit筛选id中包含cress等字符的序列
sort ${DATA_DIR}/true/cress_id.csv | uniq > ${DATA_DIR}/true/cress_id_1.csv
esearch -db protein -query "Cressdnaviricota[Organism]" | efetch -format fasta > ${DATA_DIR}/true/cress.fasta
seqkit seq -g -m 60 -M 1024 ${DATA_DIR}/true/cress.fasta > ${DATA_DIR}/true/cress_lens.fasta
seqkit seq -n ${DATA_DIR}/true/cress_lens.fasta > ${DATA_DIR}/true/cress_lens.id
grep 'rep' ${DATA_DIR}/true/cress_lens.id > ${DATA_DIR}/true/cress_lens_rep.id
#### 提取cress rep
#列匹配，a文件只有一列，在b文件整行匹配
python3 ${SCRIPTS_DIR}/列匹配.py \
  -a /home/ec2-user/project/Week3-4/data/true/cress_id_1.csv \
  -b /home/ec2-user/project/Week3-4/data/proteins_base_substituted.id \
  -o ${DATA_DIR}/true/proteins_true.id \
  #--mode extract
seqkit grep \
  -f <(cut -d ' ' -f 1 ${DATA_DIR}/true/proteins_true.id) \
  ${DATA_DIR}/true/cress_lens.fasta \
  > ${DATA_DIR}/true/proteins_true.fa
#### 聚类
cat ${DATA_DIR}/true/proteins_true.fa ${DATA_DIR}/rep/rep_clear.fa > ${DATA_DIR}/true/rep.fa
cd-hit -i ${DATA_DIR}/true/rep.fa \
        -o ${DATA_DIR}/true/rep.90.fa \
        -c 0.9 -aS 0.8 -n 5 -g 1 -G 0 -d 0 -p 1 -M 0 -T ${THREADS} 
<<EOF
file                           format  type     num_seqs    sum_len  min_len  avg_len  max_len
./Week3-4/data/true/rep.90.fa  FASTA   Protein     4,428  1,273,432       60    287.6      615
EOF

## 1.2 负样本 ${DATA_DIR}/false/false.fa
<<EOF
负样本1
${DATA_DIR}/false/rep_false.90.fa
负样本2
EOF
seqkit seq -n ${DATA_DIR}/proteins_base_substituted.fasta > ${DATA_DIR}/proteins_base_substituted.id
### 负样本 ${DATA_DIR}/false/rep_false.fa
#identify [70,] [30,70] [,30]，保留70以下所有的有相似性的序列，同时e值大于80的所有行
diamond makedb --in ${DATA_DIR}/true/rep.90.fa --db ${DATA_DIR}/true/rep.90
diamond blastp \
    --threads 4 \
    -d ${DATA_DIR}/true/rep.90 \
    -q ${DATA_DIR}/proteins_base_substituted.fasta \
    -o ${DATA_DIR}/false/proteins_false.tsv \
    --evalue 0.001 \
    --query-cover 50 \
    -f 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
    #--id 40 \
awk '$3 > 70' ${DATA_DIR}/false/proteins_false.tsv > ${DATA_DIR}/false/proteins_false-70+.tsv
awk '$3 < 70' ${DATA_DIR}/false/proteins_false.tsv > ${DATA_DIR}/false/proteins_false-70-.tsv
#加上注释id信息，根据第一列的序列id信息
python3 ${SCRIPTS_DIR}/第一列匹配.py \
        -a ${DATA_DIR}/false/proteins_false-70-.tsv \
        -b /home/ec2-user/project/Week3-4/data/proteins_base_substituted.id \
        -o ${DATA_DIR}/false/proteins_false-70-_1.tsv
#awk -F'\t' '{print $13}' ${DATA_DIR}/false/proteins_false-70-_1.tsv > ${DATA_DIR}/false/proteins_false-70-_2.tsv
python3 ${SCRIPTS_DIR}/列匹配.py \
  -a /home/ec2-user/project/Week3-4/data/true/cress_id_1.csv \
  -b ${DATA_DIR}/false/proteins_false-70-_1.tsv \
  -o ${DATA_DIR}/false/proteins_false-70-_e.tsv \
  -v -i
#基于阈值过滤 + 去重（De-duplication）
python3 ${SCRIPTS_DIR}/diamond筛选.py \
        -i ${DATA_DIR}/false/proteins_false-70-_e.tsv \
        -o ${DATA_DIR}/false/proteins_false-70-_true-0.tsv \
        -d ${DATA_DIR}/false/proteins_false-70-_验证集.tsv \
        -e 1e-100
#### 聚类
awk -F'\t' '{print $1}' ${DATA_DIR}/false/proteins_false-70-_true-0.tsv | sort | uniq > ${DATA_DIR}/false/proteins_false-70-_true-0.id
seqkit grep \
  -f <(cut -d ' ' -f 1 ${DATA_DIR}/false/proteins_false-70-_true-0.id) \
  ${DATA_DIR}/proteins_base_substituted.fasta \
  > ${DATA_DIR}/false/rep_false.fa
#grep 'QZT32273.1' ${DATA_DIR}/false/proteins_false-70-_e.tsv
<<EOF
(bio) [ec2-user@ip-172-31-42-27 project]$ seqkit stat ${DATA_DIR}/false/rep_false.fa
file                               format  type     num_seqs    sum_len  min_len  avg_len  max_len
./Week3-4/data/false/rep_false.fa  FASTA   Protein     6,883  1,981,898       62    287.9      727
EOF

### 负样本2 ${DATA_DIR}/false/false-2.fa
#正负样本合并，基本上完全相同的序列剔除，剩余的是负样本。
cat ${DATA_DIR}/true/rep.90.fa ${DATA_DIR}/false/rep_false.90.fa > ${DATA_DIR}/false/2/2.fa
diamond makedb --in ${DATA_DIR}/false/2/2.fa --db ${DATA_DIR}/false/2/2
diamond blastp \
    --threads ${THREADS} \
    -d ${DATA_DIR}/false/2/2 \
    -q ${DATA_DIR}/proteins_base_substituted.fasta \
    -o ${DATA_DIR}/false/2/2.tsv \
    --evalue 0.001 \
    --query-cover 40 \
    --id 40 \
    -f 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
<<EOF
#46563 queries aligned
39366 queries aligned. 90 80 覆盖度和相似性
EOF
awk -F'\t' '{print $1}' ${DATA_DIR}/false/2/2.tsv | sort | uniq > ${DATA_DIR}/false/2/2_1.id
seqkit grep \
  -v -f <(cut -d ' ' -f 1 ${DATA_DIR}/false/2/2_1.id) \
  ${DATA_DIR}/proteins_base_substituted.fasta \
  > ${DATA_DIR}/false/2/rep_false-2.fa
<<EOF
(BIO) lpy@administrator-Super-Server:~/hdd/myself/AI$ seqkit stat ${DATA_DIR}/false/2/rep_false-2.fa
file                                                 format  type      num_seqs      sum_len  min_len  avg_len  max_len
/home/lpy/hdd/myself/AI/data/false/2/rep_false-2.fa  FASTA   Protein  1,436,388  522,315,800       60    363.6    1,024
EOF
#### 聚类负样本后，检查ID信息 是否有cress病毒相关
cd-hit -i ${DATA_DIR}/false/2/rep_false-2.fa \
        -o ${DATA_DIR}/false/2/rep_false-2.90 \
        -c 0.9 -aS 0.8 -n 5 -g 1 -G 0 -d 0 -p 1 -M 0 -T ${THREADS}
cd-hit -i ${DATA_DIR}/false/2/rep_false-2.90 \
        -o ${DATA_DIR}/false/2/rep_false-2.60 \
        -c 0.6 -aS 0.8 -n 4 -g 1 -G 0 -d 0 -p 1 -M 0 -T ${THREADS} 
cd-hit -i ${DATA_DIR}/false/2/rep_false-2.60 \
        -o ${DATA_DIR}/false/2/rep_false-2.40 \
        -c 0.4 -aS 0.5 -n 2 -g 1 -G 0 -d 0 -p 1 -M 0 -T ${THREADS} 
seqkit seq -n ${DATA_DIR}/false/2/rep_false-2.40 | sort | uniq > ${DATA_DIR}/false/2/rep_false-2.40.id
python3 ${SCRIPTS_DIR}/列匹配.py \
  -a ${DATA_DIR}/true/cress_id_1.csv \
  -b ${DATA_DIR}/false/2/rep_false-2.40.id \
  -o ${DATA_DIR}/false/2/rep_false-2.40-1.id \
  -v -i
#### cress病毒无关，作为负样本2
seqkit grep \
  -f <(cut -d ' ' -f 1 ${DATA_DIR}/false/2/rep_false-2.40-1.id) \
  ${DATA_DIR}/false/2/rep_false-2.40 \
  > ${DATA_DIR}/false/false-2.fa
<<EOF
(BIO) lpy@administrator-Super-Server:~/hdd/myself/AI$ seqkit stat ${DATA_DIR}/false/false-2.fa
file                                           format  type     num_seqs     sum_len  min_len  avg_len  max_len
/home/lpy/hdd/myself/AI/data/false/false-2.fa  FASTA   Protein    65,196  25,148,910       65    385.7    1,024
EOF
cat ${DATA_DIR}/false/false-2.fa ${DATA_DIR}/false/rep_false.fa > ${DATA_DIR}/false/false.fa
<<EOF
(BIO) lpy@administrator-Super-Server:~/hdd/myself/AI$ seqkit stat ${DATA_DIR}/false/false.fa
file                                         format  type     num_seqs     sum_len  min_len  avg_len  max_len
/home/lpy/hdd/myself/AI/data/false/false.fa  FASTA   Protein    72,079  27,130,808       62    376.4    1,024
EOF

## 1.3 划分数据集 (Train/Test Split)
### 碱基替换后的数据集
python ${SCRIPTS_DIR}/content.py \
    -i ${DATA_DIR}/true/rep.90.fa \
    -o ${DATA_DIR}/true/rep.90_ATCG.fa \
    --strict
python ${SCRIPTS_DIR}/content.py \
    -i ${DATA_DIR}/false/false.fa \
    -o ${DATA_DIR}/false/false_ATCG.fa \
    --strict

## 1.4 划分数据集 (Train/Test Split)
#只用输入正负样本 输出分类好的数据集 自动加上pos与neg
python3 ${SCRIPTS_DIR}/data_set.py \
        --pos ${DATA_DIR}/true/rep.90_ATCG.fa \
        --neg ${DATA_DIR}/false/false_ATCG.fa \
        --outdir ${RESULTS_DIR}/data/input \
        --split-ratio 0.95 0.02 0.03
<<EOF
========================================
FINAL DATASET REPORT (Random Split)
========================================
Set        Total      Pos        Neg        Pos/Neg Ratio
-------------------------------------------------------
TRAIN      72681      4239       68442      0.06
VAL        1530       84         1446       0.06
TEST       2296       105        2191       0.05
========================================
EOF

# 2 begging model training
## 3.1 ESM-1b embedding
python 3-lora.py > ./log/lora_training.log 2>&1
python3 test.py \
--train_path ../results/data/input/train.fasta \
--val_path ../results/data/input/val.fasta \
--output_dir ../results/model \
--log_dir ../results/logs \
--model_name facebook/esm2_t33_650M_UR50D \
--max_length 1024 \
--batch_size 8 \
--grad_accum 1 \
--epochs 10 \
--learning_rate 5e-5 \
--lora_rank 8 \
--lora_alpha 32 \
--lora_dropout 0.05 > ../results/logs/lora_training.log 2>&1
#trainable params: 3,669,762 || all params: 654,712,985 || trainable%: 0.5605
python 0mail.py
<<EOF
{'eval_loss': 0.1050809845328331, 'eval_accuracy': 0.9869281045751634, 'eval_f1': 0.9865369592165963, 'eval_mcc': 0.8684903947992905, 'eval_runtime': 22.4261, 'eval_samples_per_second': 68.224, 'eval_steps_per_second': 8.561, 'epoch': 10.0}
{'train_runtime': 27015.6637, 'train_samples_per_second': 26.903, 'train_steps_per_second': 3.363, 'train_loss': 0.01569616898989431, 'epoch': 10.0}
EOF
python integrated_analysis.py \
    --mode full \
    --input_fasta ../results/data/input/test.fasta \
    --lora_model ../results/model/final_lora_model \
    --output_dir ../results/final_output \
    --base_model facebook/esm2_t33_650M_UR50D \
    --batch_size 4 \
    --max_length 1024 \
    --num_attention_plots 10 \
    --tsne_samples 4000 > ../results/logs/final_output.log 2>&1
<<EOF
训练：模型损失
模型：概率分布直方图，嵌入向量可视化，标签阈值
生物：注意力位置
EOF

<<EOF
问题：
1 训练损失
2 真实数据有误判 嵌入向量可视化发现未分开，正样本阈值太低
3 

1 序列解决：
1.1 掩盖10%左右的信息，让模型去学习
1.2 hmm找motif位置，裁剪
1.3 正样本序列太少？？？

2 模型解决：
2.1 聚焦损失函数 CrossEntropy。改用 Focal Loss，强迫模型学习预判错的的信息
2.2 

三维结构
EOF





########################## 0.1 version
# 1 数据处理
echo "Starting data processing..."
## 数据目录
DATA_DIR="./data"
SCRIPTS_DIR="./scripts"
RESULTS_DIR="./results"
THREADS=52
#数据来源：ncbi爬取的病毒基因组预测的蛋白序列
## 1.1 数据长度可视化
seqkit fx2tab -l -n ${DATA_DIR}/all_rep_proteins.fasta | cut -f2 > ${DATA_DIR}/protein_lengths.csv
### 按照 数据长，用固定区间比较好 而不是：随机长度画图
python ${SCRIPTS_DIR}/lens.py ${DATA_DIR}/protein_lengths.csv
### 偏态分布 对数 取3倍
python ${SCRIPTS_DIR}/lens_norm_plot.py ${DATA_DIR}/protein_lengths.csv
#### 可视化时，发现错误数据。放在数据集整理时处理
### 保留数据长度在 60-1024 之间的蛋白序列
seqkit seq -g -m 60 -M 1024 ${DATA_DIR}/all_rep_proteins.fasta > ${DATA_DIR}/proteins_m_M.fasta
seqkit stats ${DATA_DIR}/all_rep_proteins.fasta
<<EOF
数量还是太多
file                           format  type      num_seqs      sum_len  min_len  avg_len  max_len
./data/all_rep_proteins.fasta  FASTA   Protein  1,577,797  647,260,224        2    410.2   14,667
file                       format  type      num_seqs      sum_len  min_len  avg_len  max_len
./data/proteins_m_M.fasta  FASTA   Protein  1,482,951  536,793,919       60      362    1,024
EOF

## 2.1 碱基替换
### https://github.com/facebookresearch/esm/blob/main/esm/constants.py esm的标准token 碱基
python ${SCRIPTS_DIR}/content.py \
    -i ${DATA_DIR}/proteins_m_M.fasta \
    -o ${DATA_DIR}/proteins_base_substituted.fasta \
    --strict

<<EOF
之后处理在 美格 上跑
EOF
## 2.2 比对 ICTV 鉴定正样本
### 2.2.1 下载 标准 rep蛋白库 文广
python ${SCRIPTS_DIR}/content.py \
    -i ${DATA_DIR}/rep/rep.fa \
    -o ${DATA_DIR}/rep/rep_clear.fa \
    --strict

## 2.3 去冗余/聚类 clustering deduplication
### 2.3.1 正负样本diamond判断
diamond makedb --in ${DATA_DIR}/rep/rep_clear.fa --db ${DATA_DIR}/rep/rep_clear.dmnd
diamond blastp \
    --threads 150 \
    -d ${DATA_DIR}/rep/rep_clear.dmnd \
    -q ${DATA_DIR}/proteins_base_substituted.fasta \
    -o ${RESULTS_DIR}/data/proteins_base_substituted.csv \
    --evalue 0.001 \
    -f 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore \
    --max-target-seqs 5
#比对上的序列基本上是一样的序列

###提取比对上的序列作为正样本，并在源数据集中提取剩下的序列作为负样本
cut -f1 ${RESULTS_DIR}/data/proteins_base_substituted.csv | sort | uniq > ${RESULTS_DIR}/data/positive_ids_1.txt
seqkit grep -f ${RESULTS_DIR}/data/positive_ids_1.txt ${DATA_DIR}/proteins_base_substituted.fasta > ${RESULTS_DIR}/data/positive_samples.fasta
seqkit grep -v -f ${RESULTS_DIR}/data/positive_ids_1.txt ${DATA_DIR}/proteins_base_substituted.fasta > ${RESULTS_DIR}/data/negative_samples.fasta
<<EOF
(BIO) [EnviroDNA@compute9870 week3-4]$ seqkit stat ${RESULTS_DIR}/data/positive_samples.fasta 
file                                   format  type     num_seqs     sum_len  min_len  avg_len  max_len
./results/data/positive_samples.fasta  FASTA   Protein    43,181  13,199,314       60    305.7      898

(BIO) [EnviroDNA@compute9870 week3-4]$ seqkit stat ${RESULTS_DIR}/data/negative_samples.fasta
file                                   format  type      num_seqs      sum_len  min_len  avg_len  max_len
./results/data/negative_samples.fasta  FASTA   Protein  1,439,770  523,594,605       60    363.7    1,024
EOF
diamond blastp \
    --threads 150 \
    -d ${DATA_DIR}/rep/rep_clear.dmnd \
    -q ${RESULTS_DIR}/data/negative_samples.fasta \
    -o ${RESULTS_DIR}/data/proteins_base_substituted_2.csv \
    --evalue 10000 \
    -f 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
#结果：基本上没有比对上

### 2.3.2 正负样本挑选
#mmseq2去除完全一样的作为 实验的验证集，其他的作为正样本
#cdhit 做聚类，提取代表序列作为 模型的数据来源； 正负样本不混合聚类，按照比值聚类100:26
##去重（Identity 100% 或 99%）：去掉完全一样的序列 覆盖度 90% 相似度99%

#### 2.3.2.1 正样本
### 实验的验证集 17,410 mmseq_rep_seq.fasta
####mmseq_rep_seq.fasta easy-cluster 输入 输出临时目录 最小序列一致性0.9
seqkit replace ${RESULTS_DIR}/data/positive_samples.fasta -p '\s+.*$' -r "" -o ${RESULTS_DIR}/data/positive_samples_1.fasta
mmseqs easy-cluster ${RESULTS_DIR}/data/positive_samples_1.fasta ${RESULTS_DIR}/data/mmseq ${RESULTS_DIR}/data/tmp --min-seq-id 0.99 -c 0.9 --cov-mode 1 > ${RESULTS_DIR}/data/log.log 2>&1  #17,410 实验的验证集
seqkit seq -n ${RESULTS_DIR}/data/mmseq_rep_seq.fasta | sort | uniq > ${RESULTS_DIR}/data/mmseq_rep_seq_ids.txt # 正样本 训练集
seqkit seq -n ${RESULTS_DIR}/data/mmseq_all_seqs.fasta | sort | uniq > ${RESULTS_DIR}/data/mmseq_all_seqs.txt 

### 正样本 2,252  712,393
####提取正样本id，进行筛选 （手动）
# 原理：先把 file2 读进哈希表 总文件，再读 file1，如果不在表里就打印
awk 'NR==FNR{a[$1];next} !($1 in a)' ${RESULTS_DIR}/data/mmseq_rep_seq_ids.txt ${RESULTS_DIR}/data/mmseq_all_seqs.txt > ${RESULTS_DIR}/data/positive_true_deduped.id # 从所有序列中去掉 17410 实验验证集，剩下的作为正样本
seqkit grep -f ${RESULTS_DIR}/data/positive_true_deduped.id ${RESULTS_DIR}/data/mmseq_all_seqs.fasta > ${RESULTS_DIR}/data/positive_true_deduped.fasta
<<EOF
(BIO) [EnviroDNA@compute9870 week3-4]$ seqkit stat ${RESULTS_DIR}/data/positive_true_deduped.fasta
file                                        format  type     num_seqs    sum_len  min_len  avg_len  max_len
./results/data/positive_true_deduped.fasta  FASTA   Protein    25,771  7,887,148       62      306      749
EOF
####检查真阳性：比对到 ICTV rep 库的正样本，基本上都能够比对上
diamond blastp \
    --threads 150 \
    -d ${DATA_DIR}/rep/rep_clear.dmnd \
    -q ${RESULTS_DIR}/data/positive_true_deduped.fasta \
    --evalue 0.001

###cdhit聚类 
cd-hit -i ${RESULTS_DIR}/data/positive_true_deduped.fasta \
        -o ${RESULTS_DIR}/data/positive/pep.out.fa.90 \
        -c 0.9 -aS 0.8 -n 5 -g 1 -G 0 -d 0 -p 1 -M 0 -T ${THREADS} 
<<EOF
(BIO) [EnviroDNA@compute9870 week3-4]$ seqkit stat ${RESULTS_DIR}/data/positive/pep.out.fa.90
file                                   format  type     num_seqs  sum_len  min_len  avg_len  max_len
./results/data/positive/pep.out.fa.90  FASTA   Protein     2,252  712,393       69    316.3      749
cd-hit -i ${RESULTS_DIR}/data/positive/pep.out.fa.90 \
        -o ${RESULTS_DIR}/data/positive/pep.out.fa.60 \
        -c 0.6 -aS 0.8 -n 4 -g 1 -G 0 -d 0 -p 1 -M 0 -T ${THREADS} 
cd-hit -i ${RESULTS_DIR}/data/positive/pep.out.fa.60 \
        -o ${RESULTS_DIR}/data/positive/pep.out.fa.40 \
        -c 0.4 -aS 0.5 -n 2 -g 1 -G 0 -d 0 -p 1 -M 0 -T ${THREADS} 
EOF
#### 2.3.2.2 负样本 
##### 正样本 2,252 负样本 8,661 按照 100:26 比例
mkdir ${RESULTS_DIR}/data/negative
cd-hit -i ${RESULTS_DIR}/data/negative_samples.fasta \
        -o ${RESULTS_DIR}/data/negative/pep.out.fa.90 \
        -c 0.9 -aS 0.8 -n 5 -g 1 -G 0 -d 0 -p 1 -M 0 -T ${THREADS} 
cd-hit -i ${RESULTS_DIR}/data/negative/pep.out.fa.90 \
        -o ${RESULTS_DIR}/data/negative/pep.out.fa.60 \
        -c 0.6 -aS 0.8 -n 4 -g 1 -G 0 -d 0 -p 1 -M 0 -T ${THREADS} 
cd-hit -i ${RESULTS_DIR}/data/negative/pep.out.fa.60 \
        -o ${RESULTS_DIR}/data/negative/pep.out.fa.40 \
        -c 0.4 -aS 0.5 -n 2 -g 1 -G 0 -d 0 -p 1 -M 0 -T ${THREADS}
<<EOF
(week2) [ec2-user@ip-172-31-14-78 project]$ seqkit stat ${RESULTS_DIR}/data/negative/pep.out.fa.40
file                                   format  type     num_seqs     sum_len  min_len  avg_len  max_len
./results/data/negative/pep.out.fa.40  FASTA   Protein    65,544  25,266,949       65    385.5    1,024
EOF
##### 随机抽取 8,661 作为 负样本，
seqkit sample -n 8661 -s 42 ${RESULTS_DIR}/data/negative/pep.out.fa.40 > ${RESULTS_DIR}/data/negative/negative_samples_final.fasta
    
## 2.4 划分数据集 (Train/Test Split)
#只用输入正负样本 输出分类好的数据集 自动加上pos与neg
python ${SCRIPTS_DIR}/data_set.py \
        --pos ${RESULTS_DIR}/data/positive_true_deduped.fasta \
        --neg ${RESULTS_DIR}/data/negative/pep.out.fa.40 \
        --outdir /home/ec2-user/project/results/data/input \
        --split-ratio 0.95 0.02 0.03

# 3 begging model training
## 3.1 ESM-1b embedding
python ${SCRIPTS_DIR}/3-lora.py > ./log/lora_training.log 2>&1
python ${SCRIPTS_DIR}/0mail.py

## 3.2 test model evaluation
<<EOF
总结：两个方面：
1结果：型训练过程的loss 生信标准曲线（真实数据集） 概率分布直方图
2生物：混淆矩阵 可视化降维 三维结构
从以下几个方面验证模型的准确性：
1 loss与生信标准曲线（F1值等） 3-lora.py的图片结果分析
2 真实数据集： 4-evaluation.py + 5-true_DB.py
   真阳性：ICTV数据库中的rep蛋白序列 17,410
   真阴性：文广的cap蛋白序列 
看模型的泛化能力，以及模型关注在残基位置motif，混淆矩阵
3 模型注意力机制
   嵌入向量可视化降维
   概率分布直方图
   三维结构
EOF
### 3.2.1 loss与生信标准曲线（F1值等） 3-lora.py的图片结果分析
python ${SCRIPTS_DIR}/3-loss.py > ./log/loss_evaluation.log 2>&1
#情况：loss在epoch1直接降到0.2以下，说明模型收敛很快。而MCC和F1值在epoch1后在0.95以上。 /home/ec2-user/project/results/model/training_curves_fixed.png
#结论：rep蛋白过于简单；MCC超过0.9，不确定是否是正负集太相似
#验证：用真实数据集测试模型的泛化能力

### 3.2.2 真实数据集验证模型泛化能力 分类，混淆矩阵
#先加上pos neg标签
python ${SCRIPTS_DIR}/4-evaluation.py \
        -i ${RESULTS_DIR}/test/true_verify/data/true_verify.fasta \
        -m ./results/model/final_lora_model 
        -o ./results/4-test_predictions.csv > ./log/4-model_evaluation.log 2>&1 #output：真实样本的分类情况四列:"ID" "True_Label"         "Predicted_Label"        "Positive_Prob"
python ${SCRIPTS_DIR}/5-true_DB.py > ./log/5-true_DB_evaluation.log 2>&1 # 混淆矩阵 可视化

### 3.2.3 降维 直方图 三维结构
#太夸张，44G显存被撑爆。batch_size调小
python ${SCRIPTS_DIR}/6-statistics_biology.py \
        -i /home/ec2-user/project/results/test/true_verify/data/true_verify.fasta \
        -m ./results/model/final_lora_model \
        -o ./results/6-analysis_v2 \
        --tsne_samples 3000 \
        --batch_size 8 > ./log/6-statistics_biology.log 2>&1
#三维结构注意力映射：1 fa三维结构 输入pymol 2 脚本和txt 输入pymol 3 分析红色部分的作用：活性位点或者Domain

# 上传文件
echo "Uploading processed data to S3..."
bash ${SCRIPTS_DIR}/aws-down.sh \
    s3://ginger-ohio/Student/LPY/new_project/ \
    /home/ec2-user/project


MCC 马修斯系数 = (TP * TN - FP * FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
一般在二分类问题中使用，尤其是当类分布不均衡时。MCC的值范围从-1到+1。
一般模型范围为：0.8左右，0.9以上为优秀模型。
作用：解决二分类任务

tsne t-SNE（t-Distributed Stochastic Neighbor Embedding）是一个计算量极大的非线性算法
定义：用于高维数据的降维和可视化。它通过将高维数据点映射到低维空间（通常是二维或三维），同时尽可能保留数据点之间的局部结构关系，从而使得在低维空间中相似的数据点仍然保持接近，而不相似的数据点则被分开。
复杂度：O(N^2)或者O(N log N)

相同功能的算法：UMAP保留全局结构（适用于总体结构）  PCA主成分分析，线性降维