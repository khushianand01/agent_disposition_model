import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# =========================
# CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen3-8B"  
DATA_PATH = "data/splits/train_final.json"
VAL_DATA_PATH = "data/splits/val_final.json"
OUTPUT_DIR = "outputs/qwen3_8b_lora_mapped"
MAX_SEQ_LEN = 2048  # Qwen3 supports more, but 2048 or 4096 is good for starters
DTYPE = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
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
    r=16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0, # Supports any, but = 0 is optimized
    bias="none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth", # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # Rank stabilized LoRA
    loftq_config=None, # And LoftQ
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

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_recall_fscore_support

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Simple parsing logic (robustness needed)
    pred_dispositions = []
    pred_payment = []
    pred_reasons = []
    ptp_correct = []
    ptp_total = 0
    
    true_dispositions = []
    true_payment = []
    true_reasons = []
    
    # We need the ground truth. Since 'labels' is tokenized, we rely on the decoded text 
    # OR we need to pass the dataset references. 
    # Standard HF compute_metrics receives tokenized labels. 
    # For generation tasks, labels are the target text tokens.
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    for pred, label in zip(decoded_preds, decoded_labels):
        # Extract Output JSON part (simple split by "Response:")
        try:
            pred_json_str = pred.split("### Response:")[-1].strip()
            pred_obj = json.loads(pred_json_str) 
        except:
            pred_obj = {}
            
        try:
            label_json_str = label.split("### Response:")[-1].strip()
            label_obj = json.loads(label_json_str)
        except:
            label_obj = {} # Should not happen if data is clean

        # Extract fields
        p_disp = pred_obj.get("disposition", "UNKNOWN")
        t_disp = label_obj.get("disposition", "UNKNOWN")
        
        p_pay = pred_obj.get("payment_disposition", "UNKNOWN")
        t_pay = label_obj.get("payment_disposition", "UNKNOWN")
        
        p_reason = pred_obj.get("reason_for_not_paying", "UNKNOWN")
        t_reason = label_obj.get("reason_for_not_paying", "UNKNOWN")
        
        true_dispositions.append(t_disp)
        pred_dispositions.append(p_disp)
        
        true_payment.append(t_pay)
        pred_payment.append(p_pay)
        
        true_reasons.append(t_reason)
        pred_reasons.append(p_reason)
        
        # PTP Recall check
        # Recall = TP / (TP + FN) -> We care about cases where True was PTP
        if t_pay == "PTP":
            ptp_total += 1
            if p_pay == "PTP":
                ptp_correct.append(1)
            else:
                ptp_correct.append(0)

    # Calculate Metrics
    # Macro F1
    disp_f1 = f1_score(true_dispositions, pred_dispositions, average='macro', zero_division=0)
    pay_f1 = f1_score(true_payment, pred_payment, average='macro', zero_division=0)
    reason_f1 = f1_score(true_reasons, pred_reasons, average='macro', zero_division=0)
    
    # PTP Recall
    ptp_rec = sum(ptp_correct) / ptp_total if ptp_total > 0 else 0.0

    return {
        "disposition_macro_f1": disp_f1,
        "payment_disposition_macro_f1": pay_f1,
        "reason_macro_f1": reason_f1,
        "PTP_recall": ptp_rec
    }

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
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    # max_steps=60, # For testing purposes. 
    num_train_epochs=1, # Uncomment for full training
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    
    # Validation & Metrics
    eval_strategy="steps",
    eval_steps=500, # Validation with generation is slow, do it less frequently
    save_strategy="steps",
    save_steps=500,
    # predict_with_generate=True, # Not supported in SFTTrainer's TrainingArguments
    load_best_model_at_end=True,
    # metric_for_best_model="disposition_macro_f1", # We will rely on Loss
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
    packing=False, # Can set True for faster training
    args=training_args,
    # compute_metrics=compute_metrics, # Generation-based metrics are tricky in SFTTrainer; we use evaluate_model.py post-training
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
