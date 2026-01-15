#!/usr/bin/env python3
"""
Skeleton for LoRA fine-tuning with ESM2 sequence classification.
Fill in the TODOs to make it runnable; keep the overall flow if you want the
training curve plotting and checkpoint resume behavior.
"""
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from transformers import (
    DataCollatorWithPadding,
    EsmForSequenceClassification,
    EsmTokenizer,
    Trainer,
    TrainingArguments,
)

# =============== 1. Configuration ===============
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # TODO: choose model checkpoint
TRAIN_FASTA = "path/to/train.fasta"           # TODO: update training FASTA path
VAL_FASTA = "path/to/val.fasta"               # TODO: update validation FASTA path
OUTPUT_DIR = "./results/model"
LOG_DIR = "./results/logs"


def parse_fasta(file_path: str) -> Dataset:
    """Load FASTA and return a HF Dataset with columns sequence and label."""
    # TODO: read FASTA and produce sequences + labels
    raise NotImplementedError


def tokenize_function(tokenizer: EsmTokenizer, examples):
    """Tokenize batched sequences."""
    # TODO: map examples["sequence"] through tokenizer with truncation
    raise NotImplementedError


def build_model(model_name: str, num_labels: int = 2) -> torch.nn.Module:
    """Init base model and wrap with LoRA adapters."""
    # TODO: tweak LoRAConfig (r, alpha, target_modules, dropout, bias)
    base = EsmForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir="./model",
    )
    lora_config = LoraConfig(
        r=0,  # TODO: set rank
        lora_alpha=32,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    return get_peft_model(base, lora_config)


def compute_metrics(eval_pred):
    """Convert logits + labels into metrics dict."""
    # TODO: adjust metrics if needed
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "mcc": matthews_corrcoef(labels, preds),
        "f1": f1_score(labels, preds),
        "acc": accuracy_score(labels, preds),
        # "precision": precision_score(labels, preds),
        # "recall": recall_score(labels, preds),
    }


def create_training_args() -> TrainingArguments:
    """Central place to tune TrainingArguments."""
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        report_to="tensorboard",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=4,
        fp16=True,
        num_train_epochs=10,
        weight_decay=0.05,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
    )


def detect_last_checkpoint(output_dir: str) -> Optional[str]:
    """Return latest checkpoint path if available, else None."""
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, checkpoints[-1])


def make_trainer(model, tokenizer, train_ds, val_ds) -> Trainer:
    """Assemble Trainer with data collator and metrics."""
    args = create_training_args()
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )


def plot_history(log_history, save_path: str):
    """Plot loss/metrics history; expects Trainer.state.log_history."""
    # TODO: refine cleaning logic if you change logging
    df = pd.DataFrame(log_history)
    if len(df) > 1:
        restart_points = df[df["epoch"].diff() < 0].index.tolist()
        if restart_points:
            last_restart = restart_points[-1]
            df = df.iloc[last_restart:].reset_index(drop=True)
    train_logs = df[df["loss"].notna() & df["eval_loss"].isna()]
    eval_logs = df[df["eval_loss"].notna()]
    if train_logs.empty:
        print("No training logs; skip plotting")
        return
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_logs["epoch"], train_logs["loss"], label="Training Loss")
    if not eval_logs.empty:
        plt.plot(eval_logs["epoch"], eval_logs["eval_loss"], label="Validation Loss")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve")
    plt.subplot(1, 2, 2)
    if not eval_logs.empty:
        if "eval_f1" in eval_logs:
            plt.plot(eval_logs["epoch"], eval_logs["eval_f1"], label="F1")
        if "eval_mcc" in eval_logs:
            plt.plot(eval_logs["epoch"], eval_logs["eval_mcc"], label="MCC")
        if "eval_acc" in eval_logs:
            plt.plot(eval_logs["epoch"], eval_logs["eval_acc"], label="Accuracy")
        plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Eval Metrics")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training curves to {save_path}")


def main():
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    # TODO: preprocess datasets
    train_raw = parse_fasta(TRAIN_FASTA)
    val_raw = parse_fasta(VAL_FASTA)
    # TODO: map tokenize_function with batched=True, max_length=1024
    train_tok = train_raw
    val_tok = val_raw
    model = build_model(MODEL_NAME, num_labels=2)
    model.print_trainable_parameters()
    trainer = make_trainer(model, tokenizer, train_tok, val_tok)
    last_ckpt = detect_last_checkpoint(OUTPUT_DIR)
    trainer.train(resume_from_checkpoint=bool(last_ckpt))
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_lora_model"))
    plot_history(trainer.state.log_history, os.path.join(OUTPUT_DIR, "training_curves.png"))


if __name__ == "__main__":
    main()
