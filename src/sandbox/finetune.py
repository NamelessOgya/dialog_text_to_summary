"""
    # QLoRA Fine-tuning Example
    poetry run python -m src.sandbox.finetune
"""

import os
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import numpy as np

# 1. 設定
model_name = "rinna/gemma-2-baku-2b-it"
output_dir = "./qlora-finetuned"
batch_size = 4
micro_batch_size = 1
gradient_accumulation_steps = batch_size // micro_batch_size
max_length = 512
learning_rate = 2e-4
num_train_epochs = 100
seed = 42

# 2. データロード & 分割
data_files = {"csv": "./data/processed_data.csv"}  # カレントに data.csv がある前提
raw = load_dataset("csv", data_files=data_files)["csv"]

# 80/10/10 split
split1 = raw.train_test_split(test_size=0.2, seed=seed)
split2 = split1["test"].train_test_split(test_size=0.5, seed=seed)
dataset = DatasetDict({
    "train": split1["train"],
    "validation": split2["train"],
    "test": split2["test"],
})

# 3. トークナイザー
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id

# テキスト→モデル入力への変換関数
def preprocess(example):
    prompt = example["transcript"]
    target = example["description"]
    # 入力と出力を一続きにし、ラベルは出力部分のみ学習
    inputs = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length")
    labels = tokenizer(target, truncation=True, max_length=max_length, padding="max_length")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # 出力以外のラベルを -100 にして loss 計算から除外
    label_ids = [-100] * len(input_ids)
    # 出力部分を末尾に配置している想定
    label_ids[-len(labels["input_ids"]):] = labels["input_ids"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label_ids,
    }

tokenized = dataset.map(preprocess, batched=False)

# 4. モデル読み込み（8bit + QLoRA セットアップ）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,             # rank
    lora_alpha=32,
    lora_dropout=0.05,
    # target_modules=["q_proj", "v_proj"],  # モデルによって変更
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ],
)
model = get_peft_model(model, lora_config)

# 5. TrainingArguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=micro_batch_size,
    per_device_eval_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    logging_steps=50,
    optim="adamw_torch",
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    seed=seed,

    # early stopping
    save_strategy="steps",
)

# 6. データコラレータ
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 7. メトリクス
metrics = evaluate.load("perplexity")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # perplexity expects labels, logits
    return {"perplexity": metrics.compute(predictions=logits, references=labels)["perplexity"]}

# 8. Trainer 準備
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3,  # 評価ロスが 3 回連続で改善しなければ停止
                              early_stopping_threshold=0.0)
    ],
    # compute_metrics=compute_metrics,
)

# 9. 学習
trainer.train()

# 10. テスト評価
# --- 10. テスト評価 の部分を書き換え ---
# Trainer.evaluate() は eval_loss を含む dict を返す
results = trainer.evaluate(tokenized["test"])
eval_loss = results["eval_loss"]
perplexity = float(np.exp(eval_loss))  # numpy をインポートしておく

print("=== Test results ===")
print(f"eval_loss    : {eval_loss:.4f}")
print(f"perplexity   : {perplexity:.4f}")
