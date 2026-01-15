# 项目目标：
结合之前的javis，打造1.0版本
具体要求：
1. 优化之前分类模型，各种指标看起来健康（先分析图表，找到具体问题在哪里）
2. 优化代码，再次进行模型构建
3. 打包代码 docker yaml 

1 具体执行：
0 如何判断模型的好坏？如何优化训练集，能让模型看起来更靠谱？
AI模型验证：1 loss 生信标准曲线 2 验证集的预测比值 3 验证集的聚类情况
生物：1 嵌入向量注意力矩阵

1.1 分析图表
Week3-4/results/model/training_curves_fixed.png loss 生信标准曲线
Week3-4/results/5-inference_vis/attention_POS_WQA30011.1.png 嵌入向量注意力矩阵在真实一条序列
Week3-4/results/6-analysis_v2/prob_histogram.png 验证集的预测比值可视化 Week3-4/results/5-inference_vis/true-prediction_results.csv 验证集的预测比值
Week3-4/results/6-analysis_v2/tsne_embedding.png 验证集的聚类情况
1.2 问题根源
模型找到捷径，没有从生物学序列的方式去分析数据（判断依据：各项指标比较好，但损失下降很快）
解决方式：1 训练数据集设置：相似度划分数据集（train/val的相似度 小于40%）

筛选序列的方式：总体数据中筛选金银标准序列，提出正样本后，再进行比对筛选负样本。

正负样本怎么设置？
    正样本比对相似性高的（执行：1 下载，2 名称提取）
    负样本：相似性<70，且不是cress病毒的序列（相似度>70，假定不一定是cress，但是相似度太高。）
    （Execution: 
        Negative samples with similarity: 
            1 alignment, extract<70 identity, 2.1 threshold to remove duplicate sequences根据e值，去除e-100, cluster. 
        Negative samples without similarity: 
            alignment, de duplication, clustering） 
    正负样本设置的根本在于：让AI模型识别正负样本的差异性
    划分数据集：样本的相似性要小于一定的数值；在判定模型性能时，对不同序列的检测可以判断模型是否学会

相似性很高的序列：
    覆盖度高、相似度高的那些“非ICTV”序列，是训练集中最大的毒药。删除（比如说：比对到ictv的cress病毒序列，但是标签上没有virus字样的rep蛋白，可能是cress也可能不是，需要剔除。不然会给模型造成问题）
    另一种比如说细菌的rep蛋白，相似，作为负样本进行训练
    identify [70,] [30,70] [,30] 

数据来源,筛选条件,最终去向,角色
ICTV数据,官方列表,正样本池,金标准
NCBI 30W,相似度高 (>40%) 且 标题含病毒名,正样本池,银标准 (扩充多样性)
NCBI 30W,相似度高 (>40%) 且 标题不明/模糊,丢弃 (垃圾桶),缓冲区 (防止标错)
NCBI 30W,"相似度低 (<30%) 且 标题含 ""Rep""",负样本池,困难负样本 (训练核心)
UniRef/其他,无相似度，非病毒,负样本池,简单负样本 (背景噪声)

核废料序列展示：可以用作后期的分辨：（数据在第一次筛选负样本）
WGT79609.1 satellite replication initiator protein [Sophora alopecuroides yellow stunt alphasatellite 7e]
WGT79608.1 satellite replication initiator protein [Sophora alopecuroides yellow stunt alphasatellite 7b]
WGT79607.1 satellite replication initiator protein [Sophora alopecuroides yellow stunt alphasatellite 7a]
Alphasatellites（阿尔法卫星DNA）
它们不是独立的病毒（它们需要辅助病毒才能扩散），但它们确实编码真正的Rep蛋白（用于像CRESS病毒一样进行滚环复制）。从进化的角度看，它们和CRESS病毒是“表亲”