print("Starting execution...")
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

import argparse

# =========================
# CONFIG
# =========================
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/splits/train_v6.json")
parser.add_argument("--val_data_path", type=str, default="data/splits/val_v6.json")
parser.add_argument("--output_dir", type=str, default="outputs/qwen3_8b_v6_balanced")
parser.add_argument("--num_train_epochs", type=int, default=2)
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--load_best_model_at_end", type=str2bool, default=True)
args_cli = parser.parse_args()

MODEL_NAME = "Qwen/Qwen3-8B"  
DATA_PATH = args_cli.data_path
VAL_DATA_PATH = args_cli.val_data_path
OUTPUT_DIR = args_cli.output_dir
MAX_SEQ_LEN = 2048
DTYPE = None 
LOAD_IN_4BIT = True

# =========================
# LOAD MODEL (UNSLOTH)
# =========================
print(f"Loading model {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# Enable QLoRA / LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0, # Set to 0 to save memory and enable fast patching
    bias="none", 
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# =========================
# LOAD DATA & FORMAT
# =========================
print(f"Loading data from {DATA_PATH}...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Loading validation data from {VAL_DATA_PATH}...")
with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
    val_data = json.load(f)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token 

def format_sample(x):
    instruction = x.get('instruction', '')
    input_text = x.get('input', '')
    output_text = x.get('output', '')
    if isinstance(output_text, (dict, list)):
        output_text = json.dumps(output_text, ensure_ascii=False)
    
    text = alpaca_prompt.format(instruction, input_text, output_text) + EOS_TOKEN
    
    return {
        "text": text
    }

print("Formatting dataset...")
train_dataset = Dataset.from_list([format_sample(x) for x in raw_data])
eval_dataset = Dataset.from_list([format_sample(x) for x in val_data])

# =========================
# TRAINING
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=args_cli.per_device_train_batch_size,
    gradient_accumulation_steps=args_cli.gradient_accumulation_steps,
    warmup_ratio=0.05,
    num_train_epochs=args_cli.num_train_epochs,
    learning_rate=args_cli.learning_rate,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=50,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    
    # Validation
    eval_strategy="steps",
    eval_steps=args_cli.eval_steps,
    save_strategy="steps",
    save_steps=args_cli.save_steps,
    load_best_model_at_end=args_cli.load_best_model_at_end,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    per_device_eval_batch_size=2,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    dataset_num_proc=2,
    packing=True, # Enabled for faster training (2x-3x speedup)
    args=training_args,
)

# =========================
# EXECUTE
# =========================
print("Starting training...")

# Check for existing checkpoint
import os
from transformers.trainer_utils import get_last_checkpoint

last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    
if last_checkpoint:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer_stats = trainer.train()

print(f"Training complete. Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done.")
