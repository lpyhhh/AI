#!/usr/bin/env python

#全流程
import torch
from datasets import load_dataset
from transformers import AutoTokenizer,DataCollatorWithPadding

raw_datasets=load_dataset("glue","mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_funcation(example):
    return tokenizer(example["sentence1"],example["sentence2"],truncation=True)
tokenized_datasets=raw_datasets.map(tokenize_funcation,batched=True)
data_collator=DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments,AutoModelForSequenceClassification
training_args=TrainingArguments("test-trainer")