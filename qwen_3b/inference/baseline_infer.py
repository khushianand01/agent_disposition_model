import json
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# =========================
# CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen3-8B"
DATA_PATH = "data/splits/test.json"
OUT_PATH = "inference/baseline_predictions_500.json"

MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 512 # Increased for full JSON output
BASELINE_LIMIT = 10   # ONLY 10 SAMPLES for quick baseline check

# =========================
# LOAD TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# =========================
# LOAD MODEL (GPU ONLY, 4-BIT)
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="cuda",          # FORCE GPU
    trust_remote_code=True,
)

model.eval()
model.config.use_cache = False

# =========================
# PROMPT FORMAT
# =========================
def build_prompt(sample):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
"""

# =========================
# LOAD & LIMIT DATA
# =========================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

data = data[:BASELINE_LIMIT]   # LIMIT TO 500

print(f"Running baseline on {len(data)} samples")

results = []

# =========================
# RUN BASELINE INFERENCE
# =========================
with torch.no_grad():
    for row in tqdm(data, desc=f"Baseline inference ({len(data)} samples)"):
        prompt = build_prompt(row)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
        )

        gen_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        results.append({
            "prompt": row["input"],
            "gold": row["output"],
            "prediction": gen_text,
        })

# =========================
# SAVE OUTPUTS
# =========================
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Baseline predictions saved to {OUT_PATH}")
