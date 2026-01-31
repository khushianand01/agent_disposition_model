import torch
from unsloth import FastLanguageModel
import json

def main():
    MODEL_PATH = "/home/ubuntu/Disposition_model2-main/outputs/ringg_slm_stage1"
    MAX_SEQ_LEN = 2048
    DTYPE = None
    LOAD_IN_4BIT = True

    print(f"Loading model from {MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)

    alpaca_prompt = """### Instruction:
Analyze the following call transcript and classify the call outcome. Return ONLY valid JSON with 'disposition' and 'payment_disposition' fields.

### Input:
{}

### Response:
{}"""

    test_cases = [
        {
            "name": "Validation Sample 1 (Real PTP/Promise)",
            "transcript": "Agent: ... EMI pending ... Borrower: I will pay by 8 PM tonight. Agent: Thank you.",
            "expected": "PTP"
        },
        {
            "name": "Validation Sample (Wrong Number)",
            "transcript": "Agent: Hello. Borrower: Hello. Agent: Shankar Lal? Borrower: No, wrong number. This SIM changed.",
            "expected": "WRONG_NUMBER"
        },
        {
            "name": "Validation Sample (Denied to Pay - More Context)",
            "transcript": "Agent: Hello. I am calling from the bank regarding your loan. Borrower: Listen, I already told you I never took this loan. You guys are fraud. I will not pay a single rupee. Agent: But we have your documents. Borrower: They are fake. Go away.",
            "expected": "DENIED_TO_PAY"
        },
        {
            "name": "Hindi Sample (PTP)",
            "transcript": "Agent: Hello. Borrower: Haan ji namaskar. Agent: EMI pending hai. Borrower: Okay, main aaj raat 8 baje tak pay kar doonga.",
            "expected": "PTP"
        }
    ]

    print("\nRunning Batch Inference...\n")
    for case in test_cases:
        print(f"Test Case: {case['name']}")
        print(f"Transcript: {case['transcript']}")
        
        inputs = tokenizer(
        [
            alpaca_prompt.format(case['transcript'], "")
        ], return_tensors = "pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
        response = tokenizer.batch_decode(outputs)
        
        full_output = response[0]
        json_str = ""
        if "### Response:" in full_output:
            json_str = full_output.split("### Response:")[1].strip()
            for token in ["<|endoftext|>", "<|im_end|>", "###"]:
                json_str = json_str.split(token)[0].strip()
        
        print(f"Result: {json_str}")
        print("-" * 50)

if __name__ == "__main__":
    main()
