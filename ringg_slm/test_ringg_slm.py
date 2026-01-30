import torch
from unsloth import FastLanguageModel
import json
from datetime import date

# Reference Date
CURRENT_DATE = "2026-01-29"

model_name = "RinggAI/Transcript-Analytics-SLM1.5b"

print(f"Loading {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

def test_model(transcript):
    # RinggAI SLM 1.5B expects a instruction/input format
    # Based on the model card info:
    prompt = f"""You are a call transcript analyst. Extract the disposition and payment details.
Current Date: {CURRENT_DATE}

### Transcript:
{transcript}

### Response (JSON):
"""
    
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
    response = tokenizer.batch_decode(outputs)
    
    # Extract the JSON part from response
    try:
        json_str = response[0].split("### Response (JSON):")[1].split("}")[0] + "}"
        return json.loads(json_str)
    except Exception as e:
        return response[0]

test_cases = [
    "Agent: Payment pending hai. Customer: Main agle hafte (next week) pakka bhar dunga.",
    "Agent: Kab pay karenge? Customer: Next month ki 15 ko.",
    "Agent: Outstanding 2000 hai. Customer: Main sham ko pay kar dunga (today evening).",
    "Agent: You have a pending amount of 5000. Customer: I can only pay 2000 today. I will pay the rest next month.",
    "Agent: Hello, can I speak to Mr. Sharma? Customer: Call me back at 6 PM today, I am busy in a meeting.",
    "Agent: Your EMI is overdue. Customer: I am not interested in paying. My business is in loss.",
    "Agent: Can I speak to Rahul? Customer: This is his mother. He is not at home right now.",
    "Agent: Is this Mr. Suresh? Customer: No, you have the wrong number."
]

print("\n--- RinggAI 1.5B Test Results ---\n")
for i, t in enumerate(test_cases):
    print(f"Case {i+1}: {t}")
    result = test_model(t)
    print(f"Result: {json.dumps(result, indent=4)}\n")
