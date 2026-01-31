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
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/splits/train_v11_s2_balanced.json")
parser.add_argument("--val_data_path", type=str, default="data/splits/val_v11_s2_balanced.json")
parser.add_argument("--base_model_path", type=str, default="outputs/ringg_slm_stage1")
parser.add_argument("--output_dir", type=str, default="outputs/ringg_slm_stage2")
parser.add_argument("--num_train_epochs", type=int, default=3)
args_cli = parser.parse_args()

MAX_SEQ_LEN = 2048
DTYPE = None 
LOAD_IN_4BIT = True

# =========================
# LOAD MODEL
# =========================
print(f"Loading Base Stage-1 Model from {args_cli.base_model_path}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args_cli.base_model_path,
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
def format_sample(x):
    # Stage-2 input is a dict with transcript, disposition, payment_disposition
    input_text = json.dumps(x['input'], ensure_ascii=False)
    output_text = json.dumps(x['output'], ensure_ascii=False)
    
    prompt = f"""### Instruction:
{x['instruction']}

### Input:
{input_text}

### Response:
{output_text}"""
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
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_ratio=0.1,
    num_train_epochs=args_cli.num_train_epochs,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=20,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
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
    packing=False, # Disable packing for cleaner extraction
    args=training_args,
)

# =========================
# RUN
# =========================
print("Starting Stage 2 training...")
trainer.train()

print(f"Saving Stage 2 model to {args_cli.output_dir}...")
model.save_pretrained(args_cli.output_dir)
tokenizer.save_pretrained(args_cli.output_dir)
print("Stage 2 Training Complete!")
