import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    EsmForSequenceClassification,
    EsmTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model

# ===================== 1. 数据预处理（核心：清洗 label 和序列） =====================
def load_and_clean_data(train_csv_path, test_csv_path, max_seq_len=1024):
    """加载并清洗数据，确保 label 是 0/1 整数，序列长度 ≤ 1024"""
    # 读取训练/测试集
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # 清洗逻辑：过滤空值、超长序列、修正标签
    def clean_df(df):
        # 1. 删除 seq 或 label 为空的行
        df = df.dropna(subset=["seq", "label"])
        # 2. 过滤超长序列（计算序列长度）
        df["seq_len"] = df["seq"].apply(lambda x: len(str(x)))
        df = df[df["seq_len"] <= max_seq_len]
        # 3. 强制将 label 转为 0/1 整数（处理字符串/其他格式）
        df["label"] = df["label"].apply(lambda x: 1 if str(x).lower() in ["1", "positive"] else 0)
        df["label"] = df["label"].astype(int)  # 确保是整数类型
        # 4. 只保留必要列
        return df[["seq", "label"]]
    
    train_df = clean_df(train_df)
    test_df = clean_df(test_df)
    
    # 转为 HuggingFace Dataset 格式
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    print(f"训练集数量：{len(train_dataset)}，测试集数量：{len(test_dataset)}")
    return train_dataset, test_dataset

# ===================== 2. 加载模型和分词器 =====================
# 替换为你的 CSV 文件路径
TRAIN_CSV = "/home/ec2-user/project/AI/Study/Week1/data/train.csv"  # 需修改为实际路径
TEST_CSV = "/home/ec2-user/project/AI/Study/Week1/data/test.csv"    # 需修改为实际路径

# 加载清洗后的数据集
train_dataset, test_dataset = load_and_clean_data(TRAIN_CSV, TEST_CSV)

# 加载 ESM 分词器（强制 padding/truncation）
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
def tokenize_function(examples):
    """分词函数：统一序列长度为 1024"""
    return tokenizer(
        examples["seq"],
        truncation=True,          # 截断超长序列（双重保障）
        padding="max_length",     # 填充到 max_length
        max_length=1024,          # 与 Week1-2 要求一致
        return_attention_mask=True,
        return_tensors="pt"
    )

# 批量分词（保留 label 列）
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 格式化数据集（指定输入和标签列）
tokenized_train.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
tokenized_test.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# 加载 ESM 分类模型（2分类）
model = EsmForSequenceClassification.from_pretrained(
    "facebook/esm2_t6_8M_UR50D",
    num_labels=2,  # 二分类任务
    ignore_mismatched_sizes=True  # 忽略分类头权重不匹配的警告
)

# ===================== 3. 配置 LoRA =====================
lora_config = LoraConfig(
    r=16,               # LoRA 秩
    lora_alpha=32,      # 缩放因子
    target_modules=["query", "value"],  # ESM 模型的注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLASSIFICATION"
)

# 应用 LoRA 并打印可训练参数
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===================== 4. 训练配置 =====================
training_args = TrainingArguments(
    output_dir="./esm_lora_results",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # 根据 GPU 显存调整（L4 建议 8/16）
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",  # 替代废弃的 evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",       # 关闭 wandb 日志（避免额外依赖）
    remove_unused_columns=False,  # 关键：保留 label 列
    fp16=True,              # L4 支持混合精度训练
)

# 数据拼接器（处理批次 padding）
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===================== 5. 启动训练 =====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    processing_class=tokenizer,  # 替代废弃的 tokenizer 参数
)

print("开始 LoRA 训练...")
trainer.train()

# 保存模型
model.save_pretrained("./esm_lora_model")
tokenizer.save_pretrained("./esm_lora_tokenizer")