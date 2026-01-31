import torch
from unsloth import FastLanguageModel
import json
import sys

# =========================
# CONFIG
# =========================
MODEL_PATH = "/home/ubuntu/Disposition_model2-main/ringg_slm/outputs/ringg_slm_stage2"
MAX_SEQ_LEN = 2048
DTYPE = None
LOAD_IN_4BIT = True

alpaca_prompt = """### Instruction:
Extract structured payment-related information from the call transcript.

### Input:
{}

### Response:
{}"""

def get_model():
    print(f"Loading Ringg Stage 2 model from {MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def run_test(transcript, disposition="ANSWERED", payment_disposition="PTP"):
    model, tokenizer = get_model()
    
    input_dict = {
        "transcript": transcript,
        "disposition": disposition,
        "payment_disposition": payment_disposition,
        "current_date": "2026-01-30"
    }
    
    input_text = json.dumps(input_dict, ensure_ascii=False)
    prompt = alpaca_prompt.format(input_text, "")
    
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
    print("\n--- Ringg Stage 2 Single Inference Test ---")
    
    if len(sys.argv) > 1:
        transcript = sys.argv[1]
    else:
        print("\nEnter transcript:")
        transcript = sys.stdin.readline().strip()
        
    if not transcript:
        print("Error: No transcript entered.")
        sys.exit(1)
        
    # For Stage 2, we usually have Stage 1 outputs as context
    print("\nEnter Stage 1 Disposition (default: ANSWERED):")
    disp = sys.stdin.readline().strip() or "ANSWERED"
    
    print("Enter Stage 1 Payment Disposition (default: PTP):")
    p_disp = sys.stdin.readline().strip() or "PTP"
    
    print("\n[Running Stage 2 Extraction...]")
    result = run_test(transcript, disp, p_disp)
    
    print("\n" + "="*50)
    print(json.dumps(result, indent=4))
    print("="*50)
