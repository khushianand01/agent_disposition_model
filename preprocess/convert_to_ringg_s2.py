import json
import os

INPUT_FILE = 'data/calls_data_v11_s2_master.json'
OUTPUT_FILE = 'data/calls_data_v11_s2_ringg.json'

# Ringg Stage-2 Instruction
INSTRUCTION = """Analyze the following call transcript and extract structured information. Return ONLY valid JSON with these fields:
- disposition
- payment_disposition  
- reason_for_not_paying
- ptp_amount
- ptp_date (YYYY-MM-DD format)
- followup_date (YYYY-MM-DD format)
- remarks"""

def convert_to_ringg_s2():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        print("Please run create_master_v11_s2.py first.")
        return
    
    print(f"Loading master data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    ringg_data = []
    for item in data:
        transcript = item.get('input', '')
        output_obj = item.get('output', {})
        
        # Ringg format: instruction + input + output (JSON string)
        ringg_item = {
            "instruction": INSTRUCTION,
            "input": transcript,
            "output": json.dumps(output_obj, ensure_ascii=False)
        }
        ringg_data.append(ringg_item)
    
    print(f"Converted {len(ringg_data)} items to Ringg Stage-2 format.")
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(ringg_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_to_ringg_s2()
