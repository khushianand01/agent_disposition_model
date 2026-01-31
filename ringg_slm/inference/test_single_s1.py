import torch
from unsloth import FastLanguageModel
import json
import sys

# =========================
# CONFIG
# =========================
MODEL_PATH = "/home/ubuntu/Disposition_model2-main/ringg_slm/outputs/ringg_slm_stage1"
MAX_SEQ_LEN = 2048
DTYPE = None
LOAD_IN_4BIT = True

alpaca_prompt = """### Instruction:
Analyze the following call transcript and classify the call outcome. Return ONLY valid JSON with 'disposition' and 'payment_disposition' fields.

PAYMENT_DISPOSITION RULES:
- PTP (Promise to Pay): Customer commits to pay a specific amount on a specific date OR says "I will pay" with amount/date mentioned.
  Examples: "5 tareekh ko 4000 de dunga", "kal 2000 pay karunga", "salary aane pe pay kar dunga"
- NO_PAYMENT_COMMITMENT: Customer does NOT commit to any payment.
  Examples: "abhi nahi de sakta", "dekhte hain", "pata nahi"

### Input:
{}

### Response:
{}"""

def get_model():
    print(f"Loading Ringg Stage 1 model from {MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def run_test(transcript):
    model, tokenizer = get_model()
    
    prompt = alpaca_prompt.format(transcript, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        if "### Response:" in full_output:
            json_part = full_output.split("### Response:")[1].strip()
        else:
            json_part = full_output.strip()

        for token in ["<|im_end|>", "###", "</s>"]:
            json_part = json_part.split(token)[0].strip()

        json_start = json_part.find('{')
        json_end = json_part.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            clean_json = json_part[json_start:json_end]
            return json.loads(clean_json)
        return {"error": "No JSON found", "output": full_output}
    except Exception as e:
        return {"error": str(e), "output": full_output}

if __name__ == "__main__":
    print("\n--- Ringg Stage 1 Single Inference Test ---")
    
    if len(sys.argv) > 1:
        transcript = sys.argv[1]
    else:
        print("\nEnter transcript below (one line):")
        transcript = sys.stdin.readline().strip()
        
    if not transcript:
        print("Error: No transcript entered.")
        sys.exit(1)
        
    print("\n[Running Stage 1 Classification...]")
    result = run_test(transcript)
    
    print("\n" + "="*50)
    print(json.dumps(result, indent=4))
    print("="*50)
