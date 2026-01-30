import json
import os

INPUT_FILE = 'data/calls_data_v11_s2_master.json'
OUTPUT_FILE = 'data/calls_data_v11_s2_qwen.json'

def convert_to_qwen_s2():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        print("Please run create_master_v11_s2.py first.")
        return
    
    print(f"Loading master data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    qwen_data = []
    for item in data:
        transcript = item.get('input', '')
        output_obj = item.get('output', {})
        
        # Qwen format: input + output (dict)
        qwen_item = {
            "input": transcript,
            "output": output_obj
        }
        qwen_data.append(qwen_item)
    
    print(f"Converted {len(qwen_data)} items to Qwen Stage-2 format.")
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(qwen_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_to_qwen_s2()
