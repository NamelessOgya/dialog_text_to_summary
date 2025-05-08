"""
    poetry run python -m run.finetune
"""

import os
import csv
import time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import evaluate

# ----------------------------------------
# 1. 共通設定
# ----------------------------------------
model_name     = "rinna/gemma-2-baku-2b-it"
data_path      = "./data/processed_data.csv"
output_dir     = "./qlora-finetuned"
best_model_dir = "./qlora-finetuned/checkpoint-102"
seed           = 42
max_length     = 512
batch_size     = 16
micro_batch    = 1
grad_acc       = batch_size // micro_batch
num_epochs     = 50
learning_rate  = 2e-4
device         = "cuda" if torch.cuda.is_available() else "cpu"

SKIP_TRAIN    = False
SKIP_EVALUATE = False

# ----------------------------------------
# 2. データ読み込み＆split参照
# ----------------------------------------
raw = load_dataset("csv", data_files={"csv": data_path})["csv"]
dataset = DatasetDict({
    "train":      raw.filter(lambda x: x["split"] == "train"),
    "validation": raw.filter(lambda x: x["split"] == "valid"),
    "test":       raw.filter(lambda x: x["split"] == "test"),
})

# ----------------------------------------
# 3. トークナイザー＆コラレータ
# ----------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

def preprocess(example):
    prompt = example["transcript"]
    target = example["description"]
    inputs = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length")
    labels = tokenizer(target, truncation=True, max_length=max_length, padding="max_length")
    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    label_ids      = [-100] * len(input_ids)
    label_ids[-len(labels["input_ids"]):] = labels["input_ids"]
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         label_ids,
    }

# 前処理済みデータ
tokenized = dataset.map(preprocess, batched=False)

# ----------------------------------------
# モデル初期化（共通）
# ----------------------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=[
        "self_attn.q_proj", "self_attn.k_proj",
        "self_attn.v_proj", "self_attn.o_proj",
    ],
)

# ----------------------------------------
# トレーニング部分
# ----------------------------------------
if not SKIP_TRAIN:
    print("▶︎ Starting training...")
    model = get_peft_model(base_model, lora_config)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=micro_batch,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=num_epochs,
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
        save_strategy="steps",
    )
    metrics = evaluate.load("perplexity")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"perplexity": metrics.compute(predictions=logits, references=labels)["perplexity"]}
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print("✅ Training completed and model saved to", output_dir)
else:
    print("▶︎ SKIP_TRAIN is set, skipping training.")

# ----------------------------------------
# 評価／推論部分
# ----------------------------------------
if not SKIP_EVALUATE:
    print("▶︎ Starting evaluation/inference...")
    # ① ファインチューニング済みモデルのロード
    model = PeftModel.from_pretrained(base_model, best_model_dir)
    model.to(device)
    model.eval()

    # ② テスト損失 & パープレキシティ計算
    # ---- ここで不要な文字列カラムを除去 ----
    test_dataset = tokenized["test"].remove_columns(
        [c for c in tokenized["test"].column_names if c not in ("input_ids", "attention_mask", "labels")]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating loss"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            bs = batch["labels"].size(0)
            total_loss += loss.item() * bs
            total_samples += bs
    avg_loss = total_loss / total_samples
    perplexity = float(np.exp(avg_loss))
    print("=== Test results ===")
    print(f"eval_loss  : {avg_loss:.4f}")
    print(f"perplexity : {perplexity:.4f}")

    # ③ 推論 & CSV出力
    raw_test      = raw.filter(lambda x: x["split"] == "test")
    output_csv    = "./result/test_outputs.csv"
    max_new_tokens= 256
    start_time    = time.time()

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["transcript", "generated_text"])
        for ex in tqdm(raw_test, desc="Generating test outputs"):
            prompt = ex["transcript"]
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            gen  = full[len(prompt):].strip() if full.startswith(prompt) else full
            writer.writerow([prompt, gen])

    elapsed = time.time() - start_time
    print(f"✅ Inference outputs saved to {output_csv}")
    print(f"⏱️  Inference time: {elapsed:.2f} sec for {len(raw_test)} samples")
else:
    print("▶︎ SKIP_EVALUATE is set, skipping evaluation/inference.")
