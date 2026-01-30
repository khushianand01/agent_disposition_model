"""
Format Stage-2 Data to Training Template

Converts extracted Stage-2 candidates to training format.
"""

import json

INPUT_FILE = "data/stage2_from_stage1.json"
OUTPUT_FILE = "data/stage2_train_formatted.json"

INSTRUCTION = "Extract structured payment-related information from the call transcript."

def format_stage2_data():
    print(f"Loading candidates from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        candidates = json.load(f)
    
    print(f"Total candidates: {len(candidates)}")
    
    formatted_data = []
    
    for item in candidates:
        stage1_labels = item.get("stage1_labels", {})
        stage2_fields = item.get("stage2_fields", {})
        
        formatted_item = {
            "instruction": INSTRUCTION,
            "input": {
                "transcript": item.get("transcript", ""),
                "disposition": stage1_labels.get("disposition"),
                "payment_disposition": stage1_labels.get("payment_disposition")
            },
            "output": {
                "reason_for_not_paying": stage2_fields.get("reason_for_not_paying"),
                "ptp_amount": stage2_fields.get("ptp_amount"),
                "ptp_date": stage2_fields.get("ptp_date"),
                "followup_date": stage2_fields.get("followup_date"),
                "remarks": stage2_fields.get("remarks")
            }
        }
        
        formatted_data.append(formatted_item)
    
    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Formatted {len(formatted_data)} samples")
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Field statistics
    field_counts = {
        "reason_for_not_paying": 0,
        "ptp_amount": 0,
        "ptp_date": 0,
        "followup_date": 0,
        "remarks": 0
    }
    
    for item in formatted_data:
        for field in field_counts:
            if item["output"].get(field) is not None:
                field_counts[field] += 1
    
    print("\nField Coverage:")
    for field, count in field_counts.items():
        pct = (count / len(formatted_data) * 100) if formatted_data else 0
        print(f"  {field:30s} {count:6d} / {len(formatted_data)} ({pct:5.1f}%)")

if __name__ == "__main__":
    format_stage2_data()
