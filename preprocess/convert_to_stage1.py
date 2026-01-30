import json
import os
import random

INPUT_FILE = 'data/calls_data_v11_master.json'
OUTPUT_FILE = 'data/calls_data_v11_s1_master.json'

VALID_DISPOSITIONS = [
    "ANSWERED", "CALL_BACK_LATER", "ANSWERED_BY_FAMILY_MEMBER", 
    "CALL_DISCONNECTED_BY_CUSTOMER", "ANSWERED_DISCONNECTED", 
    "ANSWERED_VOICE_ISSUE", "SILENCE_ISSUE", "LANGUAGE_BARRIER", 
    "AUTOMATED_VOICE", "WRONG_NUMBER", "CUSTOMER_ABUSIVE", 
    "AGENT_BUSY_ON_ANOTHER_CALL", "FORWARDED_CALL", "RINGING", 
    "RINGING_DISCONNECTED", "WILL_ASK_TO_PAY", "DO_NOT_KNOW_THE_PERSON", 
    "OTHERS", "BUSY", "SWITCHED_OFF", "NOT_IN_CONTACT_ANYMORE", "CUSTOMER_PICKED",
    "NO_INCOMING_CALLS"
]

VALID_PAYMENT_DISPOSITIONS = [
    "PTP", "PAID", "SETTLEMENT", "PARTIAL_PAYMENT", 
    "NO_PAYMENT_COMMITMENT", "DENIED_TO_PAY", "NO_PROOF_GIVEN", 
    "WILL_PAY_AFTER_VISIT", "WANT_FORECLOSURE", "WANTS_TO_RENEGOTIATE_LOAN_TERMS", "DISPUTE", "None"
]

INSTRUCTION = "Analyze the following call transcript and classify the call outcome. Return ONLY valid JSON with 'disposition' and 'payment_disposition' fields."

def convert_to_stage1():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading master data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    # Capping Configuration
    CLASS_CAPS = {
        "CALL_BACK_LATER": 1000
    }
    class_counts = {}

    stage1_data = []
    # Shuffle to ensure we get a random set when capping
    random.shuffle(data)
    
    for item in data:
        transcript = item.get('input', '')
        output_obj = item.get('output', {})
        
        # Standardize labels: map null/missing to "None"
        disp = output_obj.get("disposition")
        if disp is None or str(disp).strip().lower() == 'none':
            disp = "None"
            
        p_disp = output_obj.get("payment_disposition")
        if p_disp is None or str(p_disp).strip().lower() == 'none':
            p_disp = "None"

        # Apply Capping
        if disp in CLASS_CAPS:
            count = class_counts.get(disp, 0)
            if count >= CLASS_CAPS[disp]:
                continue
            class_counts[disp] = count + 1
            
        # Extract the 2 required fields
        s1_output = {
            "disposition": disp,
            "payment_disposition": p_disp
        }
        
        # Convert output to a JSON string
        stage1_item = {
            "instruction": INSTRUCTION,
            "input": transcript,
            "output": json.dumps(s1_output, ensure_ascii=False)
        }
        stage1_data.append(stage1_item)

    # Print final counts for capped classes
    for cls, count in class_counts.items():
        print(f"Capped {cls} to {count} samples.")

    print(f"Total processed: {len(stage1_data)} items.")
    print(f"Converted {len(stage1_data)} items to Stage-1 format.")

    # Save as a single master file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(stage1_data, f, indent=2, ensure_ascii=False)

    print(f"Saved all {len(stage1_data)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_to_stage1()
