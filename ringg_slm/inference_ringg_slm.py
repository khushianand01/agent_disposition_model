import torch
from unsloth import FastLanguageModel
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Inference script for Ringg SLM Stage 1")
    parser.add_argument("--model_path", type=str, default="/home/ubuntu/Disposition_model2-main/outputs/ringg_slm_stage1", help="Path to the trained adapter")
    parser.add_argument("--transcript", type=str, help="Call transcript to analyze")
    args = parser.parse_args()

    # 1. Configuration
    MAX_SEQ_LEN = 2048
    DTYPE = None
    LOAD_IN_4BIT = True

    # 2. Load Model & Tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    # 3. Prompt Template (Must match training)
    alpaca_prompt = """### Instruction:
Analyze the following call transcript and classify the call outcome. Return ONLY valid JSON with 'disposition' and 'payment_disposition' fields.

### Input:
{}

### Response:
{}"""

    # 4. Input Transcript
    if args.transcript:
        transcript = args.transcript
    else:
        # Default test case
        transcript = """Agent: Hello, this is from the bank. Am I speaking with Mr. Sharma?
Borrower: Yes, tell me.
Agent: Your EMI of 2500 is overdue for this month. We suggest you pay it today to avoid penalties.
Borrower: Okay, I will pay it by 8 PM tonight.
Agent: Thank you, I will mark that down."""
        print("No transcript provided, using default example.")

    # 5. Inference
    print("\nRunning inference...")
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            transcript, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
    response = tokenizer.batch_decode(outputs)

    # 6. Extract and Clean Output
    full_output = response[0]
    if "### Response:" in full_output:
        # Extract everything after "### Response:"
        json_str = full_output.split("### Response:")[1].strip()
        # Remove any potential stop tokens or markers
        stop_tokens = ["<|endoftext|>", "<|im_end|>", "###"]
        for token in stop_tokens:
            json_str = json_str.split(token)[0].strip()
    else:
        json_str = full_output.strip()

    print("\n--- Model Output ---")
    print(json_str)
    print("--------------------")

    # Try to parse to verify JSON validity
    try:
        data = json.loads(json_str)
        print("\nValidated JSON structure:")
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"\nWarning: Output is not valid JSON. Error: {e}")

if __name__ == "__main__":
    main()
