import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import argparse
import os

# =========================
# CONFIG
# =========================
def str2bool(v):
    if isinstance(v, bool): return v
    return v.lower() in ('yes', 'true', 't', 'y', '1')

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../data/splits/train_v11_s1.json")
parser.add_argument("--val_data_path", type=str, default="../data/splits/val_v11_s1.json")
parser.add_argument("--output_dir", type=str, default="../outputs/ringg_slm_stage1")
parser.add_argument("--num_train_epochs", type=int, default=3)
args_cli = parser.parse_args()

MODEL_NAME = "RinggAI/Transcript-Analytics-SLM1.5b"
MAX_SEQ_LEN = 2048
DTYPE = None 
LOAD_IN_4BIT = True

# =========================
# LOAD MODEL
# =========================
print(f"Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# =========================
# DATA FORMATTING
# =========================
# Matching the prompt format used in test_ringg_slm.py
def format_sample(x):
    # Stage-1 format already includes 'instruction', 'input', 'output'
    # We use the SLM Alpaca/Instruction prompt format
    prompt = f"""### Instruction:
{x['instruction']}

### Input:
{x['input']}

### Response:
{x['output']}"""
    return {"text": prompt + tokenizer.eos_token}

print("Loading and formatting data...")
with open(args_cli.data_path, "r") as f:
    train_data = json.load(f)
with open(args_cli.val_data_path, "r") as f:
    val_data = json.load(f)

train_dataset = Dataset.from_list([format_sample(x) for x in train_data])
eval_dataset = Dataset.from_list([format_sample(x) for x in val_data])

# =========================
# TRAINING SETTINGS
# =========================
training_args = TrainingArguments(
    output_dir=args_cli.output_dir,
    per_device_train_batch_size=2, # Smaller model, can afford larger batch or lower VRAM
    gradient_accumulation_steps=8,
    warmup_ratio=0.1,
    num_train_epochs=args_cli.num_train_epochs,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=training_args,
)

# =========================
# RUN
# =========================
print("Starting training...")
trainer.train()

print(f"Saving to {args_cli.output_dir}...")
model.save_pretrained(args_cli.output_dir)
tokenizer.save_pretrained(args_cli.output_dir)
print("Done!")
